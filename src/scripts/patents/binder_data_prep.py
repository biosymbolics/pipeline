"""
Utils for transforming data into binder format
"""

import re
from typing import Optional
import polars as pl

from common.utils.file import save_json_as_file
from common.ner.binder.utils import generate_word_indices


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
    """
    formatted = (
        (
            df.filter(pl.col("indices").is_not_null())
            .groupby("publication_number")
            .agg(
                [
                    pl.col("publication_number").first().alias("id"),
                    pl.col("text").first(),
                    pl.col("indices")
                    .apply(lambda indices: sorted([idx[1][0] for idx in indices]))
                    .alias("entity_start_chars"),
                    pl.col("indices")
                    .apply(lambda indices: sorted([idx[1][1] for idx in indices]))
                    .alias("entity_end_chars"),
                    pl.struct(["domain", "indices"]).alias("entity_info"),
                ]
            )
            .filter(pl.col("entity_start_chars").is_not_null())
            .drop("publication_number")
        )
        .with_columns(
            pl.col("entity_info")
            .apply(
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
        .apply(lambda text: generate_word_indices(str(text)))
        .alias("word_indices")
    ).with_columns(
        pl.col("word_indices")
        .apply(lambda idxs: sorted([i[0] for i in idxs]))
        .alias("word_start_chars"),
        pl.col("word_indices")
        .apply(lambda idxs: sorted([i[1] for i in idxs]))
        .alias("word_end_chars"),
    )
    records = with_word_indices.to_dicts()
    save_json_as_file(records, "formatted_data.json")
    return formatted


def get_entity_indices(
    text: str, entity: str
) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Get the indices of an entity in a text

    Args:
        text (str): text to search
        entity (str): entity to search for

    Usage:
    ```
    # export stored query https://console.cloud.google.com/bigquery?sq=1056123862280:06dfa28ce71c45f19755ee3d5f631843
    export_file = "ner_training.csv"
    csv_df = pl.read_csv(export_file)
    df = csv_df.with_columns(csv_df.select(pl.struct(["text", "original_term"])
        .apply(lambda rec: get_entity_indices(rec["text"], rec["original_term"])).alias("indices")))
    ```

    To check:
        df.filter(pl.col("indices").is_not_null()).select(pl.struct(["text", "original_term", "indices"])
            .apply(lambda rec: print(rec["text"][rec["indices"][1][0]:rec["indices"][1][1]] if len(rec["indices"]) > 0 else "hi"))
            .alias("check"))

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
    token_re = re.compile(r"[\s\n]")
    words = token_re.split(text)
    entity_words = token_re.split(entity)
    start_words = [
        (idx, word)
        for idx, word in enumerate(words)
        if word.lower().startswith(entity_words[0].lower())
    ]
    for start_idx, _ in start_words:
        possible_entity_words = words[start_idx : start_idx + len(entity_words)]
        is_match = all(
            [
                word.lower().startswith(entity_word.lower())
                for entity_word, word in zip(entity_words, possible_entity_words)
            ]
        )
        if is_match:
            # copy of words inclusive of entity
            # then hack to avoid getting indexes that include EOS punctuation (e.g. we want "septic shock" not "septic shock.")
            words_inclusive = words[: start_idx + len(entity_words)]
            words_inclusive[-1] = re.sub("[.,;]$", "", words_inclusive[-1])
            start_char = sum([len(word) + 1 for word in words[:start_idx]])
            end_char = sum([len(word) + 1 for word in words_inclusive]) - 1
            return (
                (start_idx, start_idx + len(entity_words) - 1),  # word indices
                (start_char, end_char),  # char indices
            )
    return None
