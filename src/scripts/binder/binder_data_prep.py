"""
Utils for transforming data into binder format
"""

from functools import lru_cache
import re
import sys
import polars as pl
from pydash import compact
from transformers import AutoTokenizer  # type: ignore

import system

system.initialize()

from clients.low_level.postgres.postgres import PsqlDatabaseClient
from constants.core import (
    DEFAULT_BASE_NLP_MODEL,
    DEFAULT_TORCH_DEVICE,
)
from utils.file import save_json_as_file


def get_annotations():
    client = PsqlDatabaseClient()
    query = """
        SELECT
        concat(title, '\n', abstract) as text, s.publication_number, term, domain
        FROM
        (
            select publication_number, array_agg(domain) as domains
            FROM biosym_annotations group by publication_number
        ) s,
        biosym_annotations b_anns,
        applications apps
        where b_anns.publication_number = apps.publication_number
        AND s.publication_number = apps.publication_number
        and 'mechanisms' = any(s.domains)
        and 'compounds' = any(s.domains)
        and 'diseases' = any(s.domains)
        order by apps.publication_number
        limit 10
    """
    records = client.select(query)
    df = pl.DataFrame(records)
    return df


def generate_word_indices(text: str, tokenizer) -> list[tuple[int, int]]:
    """
    Generate word indices for a text

    Args:
        text (str): text to generate word indices for
    """
    tokenized = tokenizer(text)
    num_tokens = len(tokenized.tokens())
    char_spans = [tokenized.token_to_chars(ti) for ti in range(num_tokens)]
    return sorted([(cs.start, cs.end) for cs in char_spans if cs is not None])


@lru_cache
def get_entity_indices(
    text: str,
    entity: str,
    tokenizer,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Get the indices of an entity in a text

    Args:
        text (str): text to searc
        entity (str): entity to search for

    Returns:
        list[tuple[tuple[int, int], tuple[int, int]]]: list of indices of entity in text (word indices, char indices)
    """
    all_tokens = tokenizer.tokenize(text)
    entity_tokens = tokenizer.tokenize(entity)

    def get_index(i):
        if all_tokens[i : i + len(entity_tokens)] == entity_tokens:
            return (i, i + len(entity_tokens) - 1)
        return None

    word_indices = compact(
        [get_index(i) for i in range(len(all_tokens) - len(entity_tokens) + 1)]
    )
    char_indices = [(m.start(), m.end()) for m in re.finditer(re.escape(entity), text)]

    return list(zip(word_indices, char_indices))


def create_binder_data(debug: bool = False):
    """
    Create training data for binder model
    """
    annotations = get_annotations()
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_NLP_MODEL)
    df = annotations.with_columns(
        pl.struct(["text", "term"])
        .map_elements(lambda rec: get_entity_indices(rec["text"], rec["term"], tokenizer))  # type: ignore
        .alias("indices")
    )
    if debug:
        validation = (
            df.explode("indices")
            .filter(pl.col("indices").is_not_null())
            .select(
                pl.struct(["text", "term", "indices"])
                .map_elements(
                    lambda rec: print(
                        rec["text"][rec["indices"][1][0] : rec["indices"][1][1]]  # type: ignore
                        if len(rec["indices"]) > 1 and len(rec["indices"][1]) > 1  # type: ignore
                        else "No match"
                    )
                )
                .alias("check")
            )
        )
        print(validation.select("check").to_series().to_list())
    return format_into_binder(df)


def format_into_binder(df: pl.DataFrame):
    """
    Format a dataframe into the binder format, e.g.
    {
        "text": "Thyroid hormone receptors form distinct nuclear protein- dependent and independent complexes with a thyroid hormone response element.",
        "entity_types": ["protein", "DNA"],
        "entity_start_chars": [0, 100],
        "entity_end_chars": [30, 133],
        "id": "MEDLINE:91125342-0",
        "word_start_chars": [0, 8, 16, 26, 31, 40, 48, 57, 67, 71, 83, 93, 98, 100, 108, 116, 125, 132],
        "word_end_chars": [7, 15, 25, 30, 39, 47, 56, 66, 70, 82, 92, 97, 99, 107, 115, 124, 132, 133]
    }
    and saves to file.

    To split:
    ``` bash
    split_ratio=0.2
    export export_file="ner_training.csv"
    export output_file="output.jsonl"
    jq -c '.[]' ./formatted_data.json | while IFS= read -r line; do   echo "$line" >> "$output_file"; done
    export total_lines=$(cat "$output_file" | wc -l)
    export test_lines_rounded=`printf %.0f $(echo "$total_lines * $split_ratio" | bc -l)`
    split -l "$test_lines_rounded" output.jsonl
    mv xaa test.json
    mv xab dev.json
    mv xac train.json
    cat xad >> train.json
    cat xae >> train.json
    rm xad xae
    ```

    # --do_predict=true --model_name_or_path="/tmp/biosym/checkpoint-2200/pytorch_model.bin" --dataset_name=BIOSYM --output_dir=/tmp/biosym
    """
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_NLP_MODEL)

    formatted = (
        (
            df.explode("indices")
            .filter(pl.col("indices").is_not_null())
            .groupby("publication_number")
            .agg(
                [
                    pl.col("publication_number").first().alias("id"),
                    pl.col("text").first(),
                    pl.col("indices")
                    .map_elements(
                        lambda indices: sorted([idx[1][0] for idx in indices])
                    )
                    .alias("entity_start_chars"),
                    pl.col("indices")
                    .map_elements(
                        lambda indices: sorted([idx[1][1] for idx in indices])
                    )
                    .alias("entity_end_chars"),
                    pl.struct(["domain", "indices"]).alias("entity_info"),
                ]
            )
            .filter(pl.col("entity_start_chars").is_not_null())
            .drop("publication_number")
        )
        .with_columns(
            pl.col("entity_info")
            .map_elements(
                lambda recs: [
                    d
                    for _, d in sorted(
                        zip(
                            [rec["indices"][1][0] for rec in recs],
                            [rec["domain"] for rec in recs],
                        )
                    )
                ]
            )
            .alias("entity_types")
        )
        .drop("entity_info")
    )

    with_word_indices = formatted.with_columns(
        pl.col("text")
        .map_elements(lambda text: generate_word_indices(str(text), tokenizer))
        .alias("word_indices")
    ).with_columns(
        pl.col("word_indices")
        .map_elements(lambda idxs: [i[0] for i in idxs])
        .alias("word_start_chars"),
        pl.col("word_indices")
        .map_elements(lambda idxs: [i[1] for i in idxs])
        .alias("word_end_chars"),
    )
    records = with_word_indices.to_dicts()
    save_json_as_file(records, "formatted_data.json")
    return formatted


def main():
    """
    Create training data for binder model
    """
    create_binder_data()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            "Usage: python3 -m scripts.binder.binder_data_prep\nPreps data for training the binder model"
        )
        sys.exit()

    main()
