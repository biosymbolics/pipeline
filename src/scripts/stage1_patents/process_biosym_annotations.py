"""
To run after NER is complete
"""

import asyncio
import sys
import logging
from typing_extensions import NotRequired
from typing import Literal, Sequence, TypedDict
from pydash import compact
import polars as pl
from spacy.tokens import Doc


from clients.low_level.postgres import PsqlDatabaseClient as DatabaseClient
from constants.core import (
    SOURCE_BIOSYM_ANNOTATIONS_TABLE as SOURCE_TABLE,
    WORKING_BIOSYM_ANNOTATIONS_TABLE as WORKING_TABLE,
)
from constants.patterns.device import DEVICE_RES
from constants.patterns.intervention import (
    BEHAVIOR_RES,
    BIOLOGIC_BASE_TERMS,
    COMPOUND_BASE_TERMS,
    DIAGNOSTIC_RES,
    DOSAGE_FORM_RE,
    MECHANISM_BASE_TERMS,
    INTERVENTION_BASE_TERMS,
    INTERVENTION_PREFIXES,
    PROCEDURE_RES,
    PROCESS_RES,
    RESEARCH_TOOLS_RES,
    ROA_RE,
)
from core.ner.cleaning import EntityCleaner
from core.ner.spacy import Spacy
from data.domain.biomedical import (
    expand_term,
    expand_parens_term,
    remove_trailing_leading as _remove_trailing_leading,
    EXPAND_CONNECTING_RE,
    POTENTIAL_EXPANSION_MAX_TOKENS,
    TARGET_PARENS,
    REMOVAL_WORDS_POST,
    REMOVAL_WORDS_PRE,
    DELETION_TERMS,
    WordPlace,
)
from system import initialize
from utils.list import batch
from utils.re import get_or_re, get_hacky_stem_re

initialize()


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TextField = Literal["title", "abstract"]
TEXT_FIELDS: list[TextField] = ["title", "abstract"]


async def remove_substrings():
    """
    Removes substrings from annotations
    (annotations that are substrings of other annotations for that publication_number)

    Note: good to run pre-training
    """
    temp_table = "names_to_remove"
    query = rf"""
        SELECT t1.publication_number AS publication_number, t2.term AS removal_term
        FROM {WORKING_TABLE} t1
        JOIN {WORKING_TABLE} t2
        ON t1.publication_number = t2.publication_number
        WHERE t2.term<>t1.term
        AND t1.term ~* CONCAT('.*', escape_regex_chars(t2.term), '.*')
        AND length(t1.term) > length(t2.term)
        AND array_length(regexp_split_to_array(t2.term, '\s+'), 1) < 3
        ORDER BY length(t2.term) DESC
    """

    delete_query = f"""
        DELETE FROM {WORKING_TABLE}
        WHERE ARRAY[publication_number, term] IN (
            SELECT ARRAY[publication_number, removal_term]
            FROM {temp_table}
        )
    """

    logger.info("Removing substrings")
    client = DatabaseClient("patents")

    await client.create_from_select(query, temp_table)
    await client.execute_query(delete_query)
    await client.delete_table(temp_table)


TermMap = TypedDict(
    "TermMap",
    {"term": str, "cleaned_term": str, "publication_number": NotRequired[str]},
)


async def expand_annotations(
    base_terms_to_expand: list[str] = INTERVENTION_BASE_TERMS,
    prefix_terms: list[str] = INTERVENTION_PREFIXES,
):
    """
    Expands annotations in cases where NER only recognizes (say) "inhibitor" where "inhibitors of XYZ" is present.
    """
    client = DatabaseClient("patents")
    prefix_re = get_or_re(
        prefix_terms, "*", enforce_word_boundaries=True, word_boundary_char=r"\y"
    )
    terms_re = get_or_re(base_terms_to_expand)
    records = await client.select(
        rf"""
        SELECT term, concat(title, '. ', abstract) as text, app.publication_number
        FROM biosym_annotations ann, applications app
        where ann.publication_number = app.publication_number
        AND length(term) > 1
        AND term  ~* '^{prefix_re}[ ]?{terms_re}[ \.;-]*$'
        AND array_length(string_to_array(term, ' '), 1) <= {POTENTIAL_EXPANSION_MAX_TOKENS}
        AND (
            concat(title, '. ', abstract) ~* concat('.*', escape_regex_chars(term), ' {EXPAND_CONNECTING_RE}.*')
            OR
            concat(title, '. ', abstract) ~* concat('.*{TARGET_PARENS} ', escape_regex_chars(term), '.*') -- e.g. '(sstr4) agonists', which NER has a prob with
        )
        AND domain not in ('attributes', 'assignees')
        """
    )
    batched = batch(records, 50000)
    logger.info("Expanding annotations for %s records", len(records))
    return _expand_annotations(batched)


async def _expand_annotations(batched_records: Sequence[Sequence[dict]]):
    """
    Expands annotations in cases where NER only recognizes (say) "inhibitor" where "inhibitors of XYZ" is present.
    """
    logger.info("Expanding of/for annotations")

    nlp = Spacy.get_instance(disable=["ner"])

    def fix_term(record: dict, doc_map: dict[str, Doc]) -> TermMap | None:
        term = record["term"].strip(" -")
        publication_number = record["publication_number"]
        text = record["text"]
        text_doc = doc_map[publication_number]

        # check for hyphenated term edge-case
        fixed_term = expand_parens_term(record["text"], record["term"])

        if not fixed_term:
            fixed_term = expand_term(term, text, text_doc)

        if fixed_term is not None and fixed_term.lower() != record["term"].lower():
            return {
                "publication_number": publication_number,
                "term": term,
                "cleaned_term": fixed_term,
            }
        else:
            return None

    for i, records in enumerate(batched_records):
        docs = nlp.pipe([r["text"] for r in records], n_process=2)
        doc_map = dict(zip([r["publication_number"] for r in records], docs))
        logger.info(
            "Created docs for annotation expansion, batch %s (%s)", i, len(records)
        )
        fixed_terms = compact([fix_term(r, doc_map) for r in records])

        if len(fixed_terms) > 0:
            await _update_annotation_values(fixed_terms)
        else:
            logger.warning("No terms to fix for batch %s", i)


async def _update_annotation_values(term_to_fixed_term: list[TermMap]):
    client = DatabaseClient("patents")

    # check publication_number if we have it
    check_id = (
        len(term_to_fixed_term) > 0
        and term_to_fixed_term[0].get("publication_number") is not None
    )

    temp_table_name = "temp_annotations"
    await client.create_and_insert(temp_table_name, term_to_fixed_term)

    sql = f"""
        UPDATE {WORKING_TABLE}
        SET term = tt.cleaned_term
        FROM {temp_table_name} tt
        WHERE {WORKING_TABLE}.term = tt.term
        {f"AND {WORKING_TABLE}.publication_number = tt.publication_number" if check_id else ""}
    """

    await client.execute_query(sql)
    await client.delete_table(temp_table_name)


async def remove_trailing_leading(removal_terms: dict[str, WordPlace]):
    client = DatabaseClient("patents")
    records = await client.select(
        f"SELECT distinct term FROM {WORKING_TABLE} where length(term) > 1"
    )
    terms: list[str] = [r["term"] for r in records]
    cleaned_term = _remove_trailing_leading(terms, removal_terms)

    await _update_annotation_values(
        [
            {
                "term": term,
                "cleaned_term": cleaned_term,
            }
            for term, cleaned_term in zip(terms, cleaned_term)
            if cleaned_term != term
        ]
    )


async def clean_up_junk():
    """
    Remove trailing junk and silly matches
    """
    logger.info("Removing junk")

    queries = [
        # removes everything after a newline (add to EntityCleaner?)
        rf"update {WORKING_TABLE} set term= regexp_replace(term, '\.?\s*\n.*', '') where  term ~ '.*\n.*'",
        # unwrap (done in EntityCleaner too)
        f"update {WORKING_TABLE} "
        + r"set term=(REGEXP_REPLACE(term, '[)(]', '', 'g')) where term ~ '^[(][^)(]+[)]$'",
        rf"update {WORKING_TABLE} set term=(REGEXP_REPLACE(term, '^\"', '')) where term ~ '^\"'",
        # orphaned closing parens
        f"update {WORKING_TABLE} set term=(REGEXP_REPLACE(term, '[)]', '')) "
        + "where term ~ '.*[)]' and not term ~ '.*[(].*';",
        # leading/trailing whitespace (done in EntityCleaner too)
        rf"update {WORKING_TABLE} set term=trim(BOTH from term) where trim(term) <> term",
    ]
    client = DatabaseClient("patents")
    for sql in queries:
        await client.execute_query(sql)


async def remove_common_terms():
    """
    Remove common original terms
    """
    logger.info("Removing common terms")
    client = DatabaseClient("patents")
    common_terms = [
        *DELETION_TERMS,
        *INTERVENTION_BASE_TERMS,
        *EntityCleaner().removal_words,
    ]

    del_term_re = get_hacky_stem_re(common_terms)
    result = await client.select(f"select distinct term from {WORKING_TABLE}")
    terms = pl.Series([(r.get("term") or "").lower() for r in result])

    delete_terms = terms.filter(terms.str.contains(del_term_re)).to_list()
    logger.info("Found %s terms to delete from %s", len(delete_terms), del_term_re)
    logger.info("Deleting terms %s", delete_terms)

    del_query = rf"""
        delete from {WORKING_TABLE}
        where lower(term)=ANY(%s)
        or term is null
        or term = ''
        or length(trim(term)) < 3
        or (length(term) > 150 and term ~* '\y(?:and|or)\y') -- del if sentence
        or (length(term) > 150 and term ~* '.*[.;] .*') -- del if sentence
    """
    await DatabaseClient("patents").execute_query(del_query, (delete_terms,))


async def normalize_domains():
    """
    Normalizes domains
        - by rules
        - if the same term is used for multiple domains, pick the most common one
    """
    client = DatabaseClient("patents")

    compounds_re = rf".*\y{get_or_re(COMPOUND_BASE_TERMS)}.*"
    biologics_re = rf".*\y{get_or_re(BIOLOGIC_BASE_TERMS)}.*"  # TODO: break into intervention vs general biological thing
    mechanism_re = rf".*\y{get_or_re(MECHANISM_BASE_TERMS)}.*"
    device_re = get_or_re(DEVICE_RES)
    procedure_re = get_or_re(PROCEDURE_RES)
    diagnostic_re = get_or_re(DIAGNOSTIC_RES)
    research_re = get_or_re(RESEARCH_TOOLS_RES)
    behavioral_re = get_or_re(BEHAVIOR_RES)
    process_re = get_or_re(PROCESS_RES)

    queries = [
        f"update {WORKING_TABLE} set domain='compounds' where domain<>'compounds' AND term ~* '{compounds_re}$'",
        f"update {WORKING_TABLE} set domain='biologics' where domain<>'biologics' AND term ~* '{biologics_re}$'",
        f"update {WORKING_TABLE} set domain='mechanisms' where domain<>'mechanisms' AND term ~* '{mechanism_re}$'",
        f"update {WORKING_TABLE} set domain='procedures' where domain<>'procedures' AND term ~* '^{procedure_re}$'",
        f"update {WORKING_TABLE} set domain='processes' where domain<>'processes' AND term ~* '^{process_re}$'",
        f"update {WORKING_TABLE} set domain='diseases' where term ~* '(?:cancer.?|disease|disorder|syndrome|autism|condition|perforation|psoriasis|stiffness|malfunction|proliferation|carcinoma|obesity|hypertension|neurofibromatosis|tumou?r|glaucoma|virus|arthritis|seizure|bald|leukemia|huntington|osteo|melanoma|schizophrenia)s?$' and not term ~* '(?:treat(?:ing|ment|s)?|alleviat|anti|inhibit|modul|target|therapy|diagnos)' and domain<>'diseases'",
        f"update {WORKING_TABLE} set domain='research_tools' where domain<>'research_tools' AND term ~* '^{research_re}$'",
        f"update {WORKING_TABLE} set domain='behavioral_interventions' where domain<>'behavioral_interventions' AND term ~* '^{behavioral_re}$'",
        f"update {WORKING_TABLE} set domain='dosage_forms' where domain<>'dosage_forms' AND term ~* '^{DOSAGE_FORM_RE}$'",
        f"update {WORKING_TABLE} set domain='roas' where domain<>'roas' AND term ~* '^{ROA_RE}$'",
        f"update {WORKING_TABLE} set domain='devices' where domain<>'devices' AND term ~* '^{device_re}$'",
        f"update {WORKING_TABLE} set domain='diagnostics' where domain<>'diagnostics' AND term ~* '^{diagnostic_re}$'",
        f"update {WORKING_TABLE} set domain='diseases' where term ~* '.* (?:disease|disorder|syndrome|dysfunction|degenerat(?:ion|ive))s?$' and domain<>'diseases' and not term ~* '(?:compounds?|compositions?|reagent|anti|agent|immuni[zs]ing|drug for|imag|treat)'",
        f"delete from {WORKING_TABLE} ba using applications a where a.publication_number=ba.publication_number and array_to_string(ipc_codes, ',') ~* '.*C01.*' and domain='diseases' and not term ~* '(?:cancer|disease|disorder|syndrome|pain|gingivitis|poison|struvite|carcinoma|irritation|sepsis|deficiency|psoriasis|streptococcus|bleed)'",
    ]

    for sql in queries:
        await client.execute_query(sql)

    normalize_sql = f"""
        WITH ranked_domains AS (
            SELECT
                lower(term) as lot,
                domain,
                ROW_NUMBER() OVER (PARTITION BY lower(term) ORDER BY COUNT(*) DESC) as rank
            FROM {WORKING_TABLE}
            GROUP BY lower(term), domain
        )
        , max_domain AS (
            SELECT
                lot,
                domain AS new_domain
            FROM ranked_domains
            WHERE rank = 1
        )
        UPDATE {WORKING_TABLE} ut
        SET domain = md.new_domain
        FROM max_domain md
        WHERE lower(ut.term) = md.lot and ut.domain <> md.new_domain;
    """

    await client.execute_query(normalize_sql)


async def populate_working_biosym_annotations():
    """
    - Copies biosym annotations from source table
    - Performs various cleanups and deletions
    """
    client = DatabaseClient("patents")
    logger.info(
        "Copying source (%s) to working (%s) table", SOURCE_TABLE, WORKING_TABLE
    )
    await client.create_from_select(
        f"SELECT * from {SOURCE_TABLE} where domain<>'attributes'",
        WORKING_TABLE,
    )
    await client.execute_query(f"DROP TABLE IF EXISTS {WORKING_TABLE}")
    await client.execute_query(f"CREATE TABLE {WORKING_TABLE} AS TABLE {SOURCE_TABLE}")

    # add indices after initial load
    await client.create_indices(
        [
            {"table": WORKING_TABLE, "column": "publication_number"},
            {"table": WORKING_TABLE, "column": "term", "is_tgrm": True},
            {"table": WORKING_TABLE, "column": "domain"},
        ]
    )

    await clean_up_junk()

    # round 1 (leaves in stuff used by for/of)
    await remove_trailing_leading(REMOVAL_WORDS_PRE)

    # less specific terms in set with more specific terms
    # await remove_substrings()

    await expand_annotations()

    # round 2 (removes trailing "compound" etc)
    await remove_trailing_leading(REMOVAL_WORDS_POST)

    # clean up junk again (e.g. leading ws)
    # check: select * from biosym_annotations where term ~* '^[ ].*[ ]$';
    await clean_up_junk()

    # big updates are much faster w/o this index, and it isn't needed from here on out anyway
    await client.execute_query(
        """
        drop index trgm_index_biosym_annotations_term;
        drop index index_biosym_annotations_domain;
        """,
        ignore_error=True,
    )

    await remove_common_terms()  # remove one-off generic terms
    # await normalize_domains()

    await client.create_index(
        {
            "table": WORKING_TABLE,
            "column": "term",
            "is_lower": True,
        }
    )


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m scripts.stage1_patents.process_biosym_annotations
            Imports/cleans biosym_annotations (followed by a subsequent stage)
            """
        )
        sys.exit()

    asyncio.run(populate_working_biosym_annotations())
