"""
To run after NER is complete
"""
import sys
import logging
from typing_extensions import NotRequired
from typing import Literal, Sequence, TypedDict
from pydash import compact
import polars as pl
from spacy.tokens import Doc


from system import initialize

initialize()

from clients.low_level.postgres import PsqlDatabaseClient as DatabaseClient
from constants.core import (
    SOURCE_BIOSYM_ANNOTATIONS_TABLE as SOURCE_TABLE,
    WORKING_BIOSYM_ANNOTATIONS_TABLE as WORKING_TABLE,
)
from constants.patterns.intervention import (
    MECHANISM_BASE_TERMS,
    INTERVENTION_BASE_TERMS,
    INTERVENTION_PREFIXES,
)
from core.ner.cleaning import EntityCleaner
from core.ner.spacy import Spacy
from data.common.biomedical import (
    expand_term,
    expand_parens_term,
    remove_trailing_leading as _remove_trailing_leading,
    EXPAND_CONNECTING_RE,
    POTENTIAL_EXPANSION_MAX_TOKENS,
    TARGET_PARENS,
    REMOVAL_WORDS_PRE,
    REMOVAL_WORDS_POST,
)
from data.common.biomedical.types import WordPlace
from utils.list import batch
from utils.re import get_or_re

from .constants import DELETION_TERMS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TextField = Literal["title", "abstract"]
TEXT_FIELDS: list[TextField] = ["title", "abstract"]


def remove_substrings():
    """
    Removes substrings from annotations
    (annotations that are substrings of other annotations for that publication_number)

    Note: good to run pre-training
    """
    temp_table = "names_to_remove"
    query = rf"""
        SELECT t1.publication_number AS publication_number, t2.original_term AS removal_term
        FROM {WORKING_TABLE} t1
        JOIN {WORKING_TABLE} t2
        ON t1.publication_number = t2.publication_number
        WHERE t2.original_term<>t1.original_term
        AND t1.original_term ~* CONCAT('.*', escape_regex_chars(t2.original_term), '.*')
        AND length(t1.original_term) > length(t2.original_term)
        AND array_length(regexp_split_to_array(t2.original_term, '\s+'), 1) < 3
        ORDER BY length(t2.original_term) DESC
    """

    delete_query = f"""
        DELETE FROM {WORKING_TABLE}
        WHERE ARRAY[publication_number, original_term] IN (
            SELECT ARRAY[publication_number, removal_term]
            FROM {temp_table}
        )
    """

    logger.info("Removing substrings")
    client = DatabaseClient()

    client.create_from_select(query, temp_table)
    client.execute_query(delete_query)
    client.delete_table(temp_table)


TermMap = TypedDict(
    "TermMap",
    {"original_term": str, "cleaned_term": str, "publication_number": NotRequired[str]},
)


def expand_annotations(
    base_terms_to_expand: list[str] = INTERVENTION_BASE_TERMS,
    prefix_terms: list[str] = INTERVENTION_PREFIXES,
):
    """
    Expands annotations in cases where NER only recognizes (say) "inhibitor" where "inhibitors of XYZ" is present.
    """
    client = DatabaseClient()
    prefix_re = get_or_re([p + " " for p in prefix_terms], "*")
    terms_re = get_or_re([f"{t}s?" for t in base_terms_to_expand])
    records = client.select(
        rf"""
        SELECT original_term, concat(title, '. ', abstract) as text, app.publication_number
        FROM biosym_annotations ann, applications app
        where ann.publication_number = app.publication_number
        AND length(original_term) > 1
        AND original_term  ~* '^(?:{prefix_re})*{terms_re}[ ]?$'
        AND array_length(string_to_array(original_term, ' '), 1) <= {POTENTIAL_EXPANSION_MAX_TOKENS}
        AND (
            concat(title, '. ', abstract) ~* concat('.*', original_term, ' {EXPAND_CONNECTING_RE}.*')
            OR
            concat(title, '. ', abstract) ~* concat('.*{TARGET_PARENS} ', original_term, '.*') -- e.g. '(sstr4) agonists', which NER has a prob with
        )
        AND domain not in ('attributes', 'assignees')
        """
    )
    batched = batch(records, 50000)
    logger.info("Expanding annotations for %s records", len(records))
    return _expand_annotations(batched)


def _expand_annotations(batched_records: Sequence[Sequence[dict]]):
    """
    Expands annotations in cases where NER only recognizes (say) "inhibitor" where "inhibitors of XYZ" is present.
    """
    logger.info("Expanding of/for annotations")

    nlp = Spacy.get_instance(disable=["ner"])

    def fix_term(record: dict, doc_map: dict[str, Doc]) -> TermMap | None:
        original_term = record["original_term"].strip(" -")
        publication_number = record["publication_number"]
        text = record["text"]
        text_doc = doc_map[publication_number]

        # check for hyphenated term edge-case
        fixed_term = expand_parens_term(record["text"], record["original_term"])

        if not fixed_term:
            fixed_term = expand_term(original_term, text, text_doc)

        if (
            fixed_term is not None
            and fixed_term.lower() != record["original_term"].lower()
        ):
            return {
                "publication_number": publication_number,
                "original_term": original_term,
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
            _update_annotation_values(fixed_terms)
        else:
            logger.warning("No terms to fix for batch %s", i)


def _update_annotation_values(term_to_fixed_term: list[TermMap]):
    client = DatabaseClient()

    # check publication_number if we have it
    check_id = (
        len(term_to_fixed_term) > 0
        and term_to_fixed_term[0].get("publication_number") is not None
    )

    temp_table_name = "temp_annotations"
    client.create_and_insert(term_to_fixed_term, temp_table_name)

    sql = f"""
        UPDATE {WORKING_TABLE}
        SET original_term = tt.cleaned_term
        FROM {temp_table_name} tt
        WHERE {WORKING_TABLE}.original_term = tt.original_term
        {f"AND {WORKING_TABLE}.publication_number = tt.publication_number" if check_id else ""}
    """

    client.execute_query(sql)
    client.delete_table(temp_table_name)


def remove_trailing_leading(removal_terms: dict[str, WordPlace]):
    client = DatabaseClient()
    records = client.select(
        f"SELECT distinct original_term FROM {WORKING_TABLE} where length(original_term) > 1"
    )
    terms: list[str] = [r["original_term"] for r in records]
    cleaned_term = _remove_trailing_leading(terms, removal_terms)

    _update_annotation_values(
        [
            {
                "original_term": original_term,
                "cleaned_term": cleaned_term,
            }
            for original_term, cleaned_term in zip(terms, cleaned_term)
            if cleaned_term != original_term
        ]
    )


def clean_up_junk():
    """
    Remove trailing junk and silly matches
    """
    logger.info("Removing junk")

    queries = [
        # removes everything after a newline (add to EntityCleaner?)
        rf"update {WORKING_TABLE} set original_term= regexp_replace(original_term, '\.?\s*\n.*', '') where  original_term ~ '.*\n.*'",
        # unwrap (done in EntityCleaner too)
        f"update {WORKING_TABLE} "
        + r"set original_term=(REGEXP_REPLACE(original_term, '[)(]', '', 'g')) where original_term ~ '^[(][^)(]+[)]$'",
        rf"update {WORKING_TABLE} set original_term=(REGEXP_REPLACE(original_term, '^\"', '')) where original_term ~ '^\"'",
        # orphaned closing parens
        f"update {WORKING_TABLE} set original_term=(REGEXP_REPLACE(original_term, '[)]', '')) "
        + "where original_term ~ '.*[)]' and not original_term ~ '.*[(].*';",
        # leading/trailing whitespace (done in EntityCleaner too)
        rf"update {WORKING_TABLE} set original_term=trim(original_term) where trim(original_term) <> original_term",
    ]
    client = DatabaseClient()
    for sql in queries:
        client.execute_query(sql)


def fix_unmatched():
    """
    Example: 3 -d]pyrimidine derivatives -> Pyrrolo [2, 3 -d]pyrimidine derivatives
    """

    logger.info("Fixing unmatched parens")

    def get_query(field, char_set):
        sql = f"""
            UPDATE {WORKING_TABLE} ab
            set original_term=substring(a.{field}, CONCAT('(?i)([^ ]*{char_set[0]}.*', escape_regex_chars(original_term), ')'))
            from applications a
            WHERE ab.publication_number=a.publication_number
            AND substring(a.{field}, CONCAT('(?i)([^ ]*{char_set[0]}.*', escape_regex_chars(original_term), ')')) is not null
            AND original_term ~* '.*{char_set[1]}.*' AND not original_term ~* '.*{char_set[0]}.*'
            AND {field} ~* '.*{char_set[0]}.*{char_set[1]}.*'
        """
        return sql

    client = DatabaseClient()
    for field in TEXT_FIELDS:
        for char_set in [(r"\[", r"\]"), (r"\(", r"\)")]:
            sql = get_query(field, char_set)
            client.execute_query(sql)


def remove_common_terms():
    """
    Remove common original terms
    """
    logger.info("Removing common terms")
    client = DatabaseClient()
    common_terms = [
        *DELETION_TERMS,
        *INTERVENTION_BASE_TERMS,
        *EntityCleaner().common_words,
    ]

    common_terms_re = get_or_re(common_terms)
    del_term_res = [
        # .? - to capture things like "gripping" from "grip"
        f"^{common_terms_re}.?(?:ing|e|ied|ed|er|or|en|ion|ist|ly|able|ive|al|ic|ous|y|ate|at|ry|y|ie)*s?$",
    ]
    del_term_re = "(?i)" + get_or_re(del_term_res)
    result = client.select(f"select distinct original_term from {WORKING_TABLE}")
    terms = pl.Series([(r.get("original_term") or "").lower() for r in result])

    delete_terms = terms.filter(terms.str.contains(del_term_re)).to_list()
    logger.info("Found %s terms to delete from %s", len(delete_terms), del_term_re)
    logger.info("Deleting terms %s", delete_terms)

    del_query = rf"""
        delete from {WORKING_TABLE}
        where lower(original_term)=ANY(%s)
        or original_term is null
        or original_term = ''
        or length(trim(original_term)) < 3
        or (length(original_term) > 150 and original_term ~* '\y(?:and|or)\y') -- del if sentence
        or (length(original_term) > 150 and original_term ~* '.*[.;] .*') -- del if sentence
    """
    DatabaseClient().execute_query(del_query, (delete_terms,))


def normalize_domains():
    """
    Normalizes domains
        - by rules
        - if the same term is used for multiple domains, pick the most common one
    """
    client = DatabaseClient()

    mechanism_terms = [f"{t}s?" for t in MECHANISM_BASE_TERMS]
    mechanism_re = get_or_re(mechanism_terms)

    queries = [
        f"update {WORKING_TABLE} set domain='mechanisms' where domain<>'mechanisms' AND original_term ~* '.*{mechanism_re}$'",
        f"update {WORKING_TABLE} set domain='mechanisms' where domain<>'mechanisms' AND original_term in ('abrasive', 'dyeing', 'dialyzer', 'colorant', 'herbicidal', 'fungicidal', 'deodorant', 'chemotherapeutic',  'photodynamic', 'anticancer', 'anti-cancer', 'tumor infiltrating lymphocytes', 'electroporation', 'vibration', 'disinfecting', 'disinfection', 'gene editing', 'ultrafiltration', 'cytotoxic', 'amphiphilic', 'transfection', 'chemotherapy')",
        f"update {WORKING_TABLE} set domain='diseases' where original_term in ('adrenoleukodystrophy', 'stents') or original_term ~ '.* diseases?$'",
        f"update {WORKING_TABLE} set domain='compounds' where original_term in ('ethanol', 'isocyanates')",
        f"update {WORKING_TABLE} set domain='compounds' where original_term ~* '(?:^| |,)(?:molecules?|molecules? bindings?|reagents?|derivatives?|compositions?|compounds?|formulations?|stereoisomers?|analogs?|analogues?|homologues?|drugs?|regimens?|clones?|particles?|nanoparticles?|microparticles?)$' and not original_term ~* '(anti|receptor|degrade|disease|syndrome|condition)' and domain<>'compounds'",
        f"update {WORKING_TABLE} set domain='mechanisms' where original_term ~* '.*receptor$' and domain='compounds'",
        f"update {WORKING_TABLE} set domain='diseases' where original_term ~* '(?:cancer|disease|disorder|syndrome|autism|condition|psoriasis|carcinoma|obesity|hypertension|neurofibromatosis|tumor|tumour|glaucoma|arthritis|seizure|bald|leukemia|huntington|osteo|melanoma|schizophrenia)s?$' and not original_term ~* '(?:treat(?:ing|ment|s)?|alleviat|anti|inhibit|modul|target|therapy|diagnos)' and domain<>'diseases'",
        f"update {WORKING_TABLE} set domain='mechanisms' where original_term ~* '.*gene$' and domain='diseases' and not original_term ~* '(?:cancer|disease|disorder|syndrome|autism|associated|condition|psoriasis|carcinoma|obesity|hypertension|neurofibromatosis|tumor|tumour|glaucoma|retardation|arthritis|tosis|motor|seizure|bald|leukemia|huntington|osteo|atop|melanoma|schizophrenia|susceptibility|toma)'",
        f"update {WORKING_TABLE} set domain='mechanisms' where original_term ~* '.* factor$' and not original_term ~* '.*(?:risk|disease).*' and domain='diseases'",
        f"update {WORKING_TABLE} set domain='mechanisms' where original_term ~* 'receptors?$' and domain='diseases'",
        f"delete from {WORKING_TABLE} ba using applications a where a.publication_number=ba.publication_number and array_to_string(ipc_codes, ',') ~* '.*C01.*' and domain='diseases' and not original_term ~* '(?:cancer|disease|disorder|syndrome|pain|gingivitis|poison|struvite|carcinoma|irritation|sepsis|deficiency|psoriasis|streptococcus|bleed)'",
    ]

    for sql in queries:
        client.execute_query(sql)

    normalize_sql = f"""
        WITH ranked_domains AS (
            SELECT
                lower(original_term) as lot,
                domain,
                ROW_NUMBER() OVER (PARTITION BY lower(original_term) ORDER BY COUNT(*) DESC) as rank
            FROM {WORKING_TABLE}
            GROUP BY lower(original_term), domain
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
        WHERE lower(ut.original_term) = md.lot and ut.domain <> md.new_domain;
    """

    client.execute_query(normalize_sql)


def populate_working_biosym_annotations():
    """
    - Copies biosym annotations from source table
    - Performs various cleanups and deletions
    """
    client = DatabaseClient()
    logger.info(
        "Copying source (%s) to working (%s) table", SOURCE_TABLE, WORKING_TABLE
    )
    client.create_from_select(
        f"SELECT * from {SOURCE_TABLE} where domain<>'attributes'",
        WORKING_TABLE,
    )

    # add indices after initial load
    client.create_indices(
        [
            {"table": WORKING_TABLE, "column": "publication_number"},
            {"table": WORKING_TABLE, "column": "original_term", "is_tgrm": True},
            {"table": WORKING_TABLE, "column": "domain"},
        ]
    )

    fix_unmatched()
    clean_up_junk()

    # # round 1 (leaves in stuff used by for/of)
    remove_trailing_leading(REMOVAL_WORDS_PRE)

    remove_substrings()  # less specific terms in set with more specific terms # keeping substrings until we have ancestor search
    # after remove_substrings to avoid expanding substrings into something (potentially) mangled
    expand_annotations()

    # round 2 (removes trailing "compound" etc)
    remove_trailing_leading(REMOVAL_WORDS_POST)

    # clean up junk again (e.g. leading ws)
    # check: select * from biosym_annotations where original_term ~* '^[ ].*[ ]$';
    clean_up_junk()

    # big updates are much faster w/o this index, and it isn't needed from here on out anyway
    client.execute_query(
        """
        drop index trgm_index_biosym_annotations_original_term;
        drop index index_biosym_annotations_domain;
        """,
        ignore_error=True,
    )

    remove_common_terms()  # remove one-off generic terms

    normalize_domains()

    # do this last to minimize mucking with attribute annotations
    client.select_insert_into_table(
        f"SELECT * from {SOURCE_TABLE} where domain='attributes'", WORKING_TABLE
    )


if __name__ == "__main__":
    """
    Checks:

    select original_term, count(*) from biosym_annotations group by original_term order by count(*) desc limit 2000;
    select sum(count) from (select count(*) as count from biosym_annotations where domain not in ('attributes', 'assignees', 'inventors') and original_term<>'' group by lower(original_term) order by count(*) desc limit 1000) s;
    (556,711 -> 567,398 -> 908,930 -> 1,037,828 -> 777,772)
    select sum(count) from (select count(*) as count from biosym_annotations where domain not in ('attributes', 'assignees', 'inventors') and original_term<>'' group by lower(original_term) order by count(*) desc offset 10000) s;
    (2,555,158 -> 2,539,723 -> 3,697,848 -> 5,302,138 -> 4,866,248)
    select count(*) from biosym_annotations where domain not in ('attributes', 'assignees', 'inventors') and original_term<>'' and array_length(regexp_split_to_array(original_term, ' '), 1) > 1;
    (2,812,965 -> 2,786,428 -> 4,405,141 -> 5,918,690 -> 5,445,856)
    select count(*) from biosym_annotations where domain not in ('attributes', 'assignees', 'inventors') and original_term<>'';
    (3,748,417 -> 3,748,417 -> 5,552,648 -> 7,643,403 -> 6,749,193)
    select domain, count(*) from biosym_annotations group by domain;
    attributes | 3721861
    compounds  | 2572389
    diseases   |  845771
    mechanisms | 4225243
    ------------+---------
    attributes | 3721861
    compounds  | 2332852
    diseases   |  810242
    mechanisms | 3606099
    select sum(count) from (select original_term, count(*)  as count from biosym_annotations where original_term ilike '%inhibit%' group by original_term order by count(*) desc limit 100) s;
    (14,910 -> 15,206 -> 37,283 -> 34,083 -> 25,239 -> 22,493 -> 21,758)
    select sum(count) from (select original_term, count(*)  as count from biosym_annotations where original_term ilike '%inhibit%' group by original_term order by count(*) desc limit 1000) s;
    (38,315 -> 39,039 -> 76,872 -> 74,050 -> 59,714 -> 54,696 -> 55,104)
    select sum(count) from (select original_term, count(*)  as count from biosym_annotations where original_term ilike '%inhibit%' group by original_term order by count(*) desc offset 1000) s;
    (70,439 -> 69,715 -> 103,874 -> 165,806 -> 138,019 -> 118,443 -> 119,331)
    """
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m scripts.patents.psql.biosym_annotations
            Imports/cleans biosym_annotations (followed by a subsequent stage)
            """
        )
        sys.exit()

    populate_working_biosym_annotations()
