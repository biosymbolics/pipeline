"""
To run after NER is complete
"""
import sys
import logging
from typing import Literal

from pydash import flatten

from system import initialize

initialize()

from clients.low_level.big_query import (
    delete_bg_table,
    execute_bg_query,
    BQ_DATASET_ID,
    query_to_bg_table,
)

from ._constants import (
    INTERVENTION_BASE_TERMS,
    INTERVENTION_BASE_TERM_SETS,
    INTERVENTION_BASE_PREFIXES,
)

TextField = Literal["title", "abstract"]

WORKING_TABLE = f"{BQ_DATASET_ID}.biosym_annotations"
SOURCE_TABLE = f"{BQ_DATASET_ID}.biosym_annotations_source"
TEXT_FIELDS: list[TextField] = ["title", "abstract"]


def remove_substrings():
    """
    Removes substrings from annotations
    """
    query = f"""
        CREATE OR REPLACE TABLE {BQ_DATASET_ID}.names_to_remove AS
            SELECT t1.publication_number AS publication_number, t2.original_term AS removal_term
            FROM {WORKING_TABLE} t1
            JOIN {WORKING_TABLE} t2
            ON t1.publication_number = t2.publication_number
            WHERE t2.original_term<>t1.original_term
            AND lower(t1.original_term) like CONCAT('%', lower(t2.original_term), '%')
            AND length(t1.original_term) > length(t2.original_term)
            AND array_length(SPLIT(t2.original_term, ' ')) < 3
            ORDER BY length(t2.original_term) DESC
    """

    delete_query = f"""
        DELETE FROM {WORKING_TABLE}
        WHERE (publication_number, original_term) IN (
            SELECT (publication_number, removal_term)
            FROM {BQ_DATASET_ID}.names_to_remove
        )
    """

    execute_bg_query(query)
    execute_bg_query(delete_query)
    delete_bg_table("names_to_remove")


def fix_of_for_annotations():
    """
    Handles "inhibitors of XYZ" and the like, which neither GPT or SpaCyNER were good at finding
    (but high hopes for binder)
    """
    # Define terms

    terms = INTERVENTION_BASE_TERMS
    term_sets = INTERVENTION_BASE_TERM_SETS
    prefixes = INTERVENTION_BASE_PREFIXES

    prefix_re = "|".join([p + " " for p in prefixes])

    def get_query(term_or_term_set: str | list[str], field: TextField):
        if isinstance(term_or_term_set, list):
            # term set
            re_term = "(?:" + "|".join([f"{ts}s?" for ts in term_or_term_set]) + ")"
        else:
            re_term = term + "s?"
        sql = f"""
            UPDATE {WORKING_TABLE} a
            SET original_term=(REGEXP_EXTRACT({field}, '(?i)((?:{prefix_re})*{re_term} (?:of |for |the |that |to |comprising )+ (?:(?:the|a) )?.*?)(?:and|useful|for|,|$)'))
            FROM `fair-abbey-386416.patents.gpr_publications` p
            WHERE p.publication_number=a.publication_number
            AND REGEXP_CONTAINS(original_term, "^(?i)(?:{prefix_re})*{re_term}$")
            AND REGEXP_CONTAINS(p.{field}, '(?i).*{re_term} (?:of|for|the|that|to|comprising).*')
        """
        return sql

    def get_hyphen_query(term, field: TextField):
        re_term = term + "s?"
        sql = f"""
            UPDATE {WORKING_TABLE} a
            SET original_term=(REGEXP_EXTRACT(title, '(?i)([A-Za-z0-9]+-{re_term})'))
            FROM `fair-abbey-386416.patents.gpr_publications` p
            WHERE p.publication_number=a.publication_number
            AND REGEXP_CONTAINS(original_term, '^(?i){re_term}$')
            AND REGEXP_CONTAINS(p.{field}, '(?i).*[A-Za-z0-9]+-{re_term}.*')
        """
        return sql

    for term in terms:
        for field in TEXT_FIELDS:
            sql = get_query(term, field)
            execute_bg_query(sql)

    for term in [*terms, *[t for term_set in term_sets for t in term_set]]:
        for field in TEXT_FIELDS:
            sql = get_hyphen_query(term, field)
            execute_bg_query(sql)

    # loop over term sets, in which the original_term may be in another form than the title variant
    for term_set in term_sets:
        for field in TEXT_FIELDS:
            sql = get_query(term_set, field)
            execute_bg_query(sql)


WordPlace = Literal["leading", "trailing", "all"]


def remove_junk():
    """
    Remove trailing junk and silly matches
    """

    def get_remove_word(word, place: WordPlace):
        if place == "trailing":
            return f"""update `{WORKING_TABLE}` set original_term=(REGEXP_REPLACE(original_term, '(?i) {word}$', '')) where regexp_contains(original_term, '(?i).* {word}$')"""
        elif place == "leading":
            return f"""update `{WORKING_TABLE}` set original_term=(REGEXP_REPLACE(original_term, '(?i)^{word} ', '')) where regexp_contains(original_term, '(?i)^{word} .*')"""
        else:
            return rf"""update `{WORKING_TABLE}` set original_term=(REGEXP_REPLACE(original_term, '(?i)(?:^|$| ){word}(?:^|$| )', ' ')) where regexp_contains(original_term, '(?i)(?:^|$| ){word}(?:^|$| )')"""

    removal_words: dict[str, WordPlace] = {
        "such": "all",
        "methods?": "all",
        "obtainable": "all",
        "the": "leading",
        "excellent": "all",
        "particular": "leading",
        "useful": "trailing",
        "thereof": "trailing",
        "capable": "trailing",
        "specific": "leading",
        "novel": "leading",
        "new": "leading",
        "inventive": "leading",
        "other": "leading",
        "of": "trailing",
        "therapeutically": "trailing",
        "suitable": "all",
        "therapeutic": "leading",
        "patient": "leading",
        "patient": "trailing",
        "acceptable": "all",
        "thereto": "trailing",
        "certain": "leading",
        ", and": "trailing",
        "therapeutic procedures": "all",
        "therapeutic procedure": "all",
    }
    delete_terms = [
        "wherein a",
        "pharmaceutical compositions",
        "compound I",
        "wherein said compound",
        "fragment of",
        "pharmacological compositions",
        "therapeutically",
        "general formula (I)",
        "medicine Compounds",
        "receptacle",
        "liver",
        "treatment regimens",
        "unsubstituted",
        "Compound I",
        "medicinal compositions",
        "COMPOUND",
        "DISEASE",
        "medicine Compounds of formula",
        "THERAPY",
        "geographic locations",
        "quantitation",
        "dosage regimen",
        "administrative procedure",
        "cannula",
        "endoscope",
        "optionally substituted",
        "stent",
        "capacitor",
        "mass spectrometry",
        "suction",
        "accelerometer",
    ]
    queries = [
        *[get_remove_word(word, place) for word, place in removal_words.items()],
        f"delete from `{WORKING_TABLE}` where original_term is null",
        f"update `{WORKING_TABLE}` "
        + "set original_term=(REGEXP_REPLACE(original_term, '[ ]{2,}', ' ')) where regexp_contains(original_term, '[ ]{2,}')",
        f"update `{WORKING_TABLE}` set original_term=(REGEXP_REPLACE(original_term, '^[ ]+', '')) where regexp_contains(original_term, '^[ ]+')",
        f"update `{WORKING_TABLE}` set original_term=(REGEXP_REPLACE(original_term, '[)]', '')) where original_term like '%)' and original_term not like '%(%';",
        f"delete from `{WORKING_TABLE}` where original_term='COMPOSITION' or original_term='therapeutical' or original_term like '%transducer%' or original_term='prognosticating' or original_term in ({delete_terms}) or original_term like '% administration' or original_term like '% patients' or original_term like 'treat %' or original_term like 'treating %' or original_term like 'field of %'",
        f"update `{WORKING_TABLE}` set domain='mechanisms' where original_term in ('tumor infiltrating lymphocytes')",
        f"update `{WORKING_TABLE}` set domain='diseases' where original_term in ('adrenoleukodystrophy')",
        f"update `{WORKING_TABLE}` set original_term=(REGEXP_REPLACE(original_term, 'disease factor', 'disease')) where original_term like '% disease factor';",
        f"update `{WORKING_TABLE}` set "
        + "original_term=regexp_extract(original_term, '(.{10,})(?:\\. [A-Z]\\w{3,}).*') where regexp_contains(original_term, '.{10,}\\. [A-Z]\\w{3,}')",
        f"delete FROM `{WORKING_TABLE}` where length(original_term) < 2",
        f"update `{WORKING_TABLE}` set domain='mechanisms' where original_term like '% gene' and domain='compounds'",
        f"update `{WORKING_TABLE}` set domain='mechanisms' where original_term like '% gene' and domain='diseases' and not regexp_contains(original_term, '(?i)(?:cancer|disease|disorder|syndrome|autism|associated|condition|psoriasis|carcinoma|obesity|hypertension|neurofibromatosis|tumor|tumour|glaucoma|retardation|arthritis|tosis|motor|seizure|bald|leukemia|huntington|osteo|atop|melanoma|schizophrenia|susceptibility|toma)')",
        f"update `{WORKING_TABLE}` set domain='mechanisms' where original_term like '% factor' and original_term not like '%risk%' and original_term not like '%disease%' and domain='diseases'",
        f"update `{WORKING_TABLE}` set domain='mechanisms' where regexp_contains(original_term, 'receptors?$') and domain='diseases'",
    ]
    for sql in queries:
        results = execute_bg_query(sql)


def fix_unmatched():
    """
    Example: 3 -d]pyrimidine derivatives -> Pyrrolo [2, 3 -d]pyrimidine derivatives
    """

    def get_query(field, char_set):
        sql = rf"""
            UPDATE {WORKING_TABLE} a
            set original_term=REGEXP_EXTRACT(p.{field}, CONCAT(r'(?i)([^ ]*\{char_set[0]}.*', `{BQ_DATASET_ID}.escape_regex_chars`(original_term), ')'))
            from `{BQ_DATASET_ID}.gpr_publications` p
            WHERE p.publication_number=a.publication_number
            AND REGEXP_EXTRACT(p.{field}, CONCAT(r'(?i)([^ ]*\{char_set[0]}.*', `{BQ_DATASET_ID}.escape_regex_chars`(original_term), ')')) is not null
            AND original_term like '%{char_set[1]}%' AND original_term not like '%{char_set[0]}%'
            AND {field} like '%{char_set[0]}%{char_set[1]}%'
        """
        return sql

    for field in TEXT_FIELDS:
        for char_set in [("{", "}"), ("[", "]"), ("(", ")")]:
            sql = get_query(field, char_set)
            execute_bg_query(sql)


def remove_common_terms():
    """
    Remove common original terms
    """
    common_terms = [
        *flatten(INTERVENTION_BASE_TERM_SETS),
        *INTERVENTION_BASE_TERMS,
    ]  # regex in here, effectively ignored
    with_plurals = [
        *common_terms,
        *[f"{term}s" for term in common_terms],
    ]  # add plurals

    str_match = ", ".join(
        [f"'{term.lower()}'" for term in with_plurals if "?" not in term]
    )  # hack regex check
    re_match = " OR ".join(
        [
            f"regexp_contains(original_term, '^{term.lower()}s?$')"
            for term in common_terms
            if "?" in term
        ]
    )
    query = f"delete from {WORKING_TABLE} where lower(original_term) in ({str_match}) OR {re_match}"
    execute_bg_query(query)


if __name__ == "__main__":
    if "-h" in sys.argv:
        print("Usage: python3 ner.py\nCleans up annotations after NER is complete")
        sys.exit()

    # copy to destination table
    logging.info(
        "Copying source (%s) to working (%s) table", SOURCE_TABLE, WORKING_TABLE
    )
    query_to_bg_table("SELECT * from `{SOURCE_TABLE}`", WORKING_TABLE)

    fix_of_for_annotations()
    fix_unmatched()

    remove_junk()
    remove_substrings()
    remove_common_terms()  # final step - remove one-off generic terms
