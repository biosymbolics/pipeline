"""
To run after NER is complete
"""
import sys
import logging
from typing import Literal, Union

from system import initialize

initialize()

from clients.low_level.big_query import delete_bg_table, execute_bg_query, BQ_DATASET_ID

TextField = Literal["title", "abstract"]

TABLE = f"{BQ_DATASET_ID}.biosym_annotations"
TEXT_FIELDS: list[TextField] = ["title", "abstract"]


def remove_substrings():
    """
    Removes substrings from annotations
    """
    query = f"""
        CREATE OR REPLACE TABLE {BQ_DATASET_ID}.names_to_remove AS
            SELECT t1.publication_number AS publication_number, t2.original_term AS removal_term
            FROM {TABLE} t1
            JOIN {TABLE} t2
            ON t1.publication_number = t2.publication_number
            WHERE t2.original_term<>t1.original_term
            AND lower(t1.original_term) like CONCAT('%', lower(t2.original_term), '%')
            AND length(t1.original_term) > length(t2.original_term)
            AND array_length(SPLIT(t2.original_term, ' ')) < 3
            ORDER BY length(t2.original_term) DESC
    """

    delete_query = f"""
        DELETE FROM {TABLE}
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
    terms = [
        "blockers",
        "blockade",
        "blocking",
        "reagents",
        "derivatives",
        "antigens",
        "neoantigens",
        "compositions",
        "compounds",
        "formulations",
        "linkers",
        "isoforms",
        "stereoisomers",
        "surfactants",
        "antibodies",
        "antibody",
        "antibody molecules",
        "autoantibody",
        "autoantibodies",
        "donors",
        "prodrugs",
        "pro-drugs",
        "adjuvants",
        "vaccines",
        "vaccine adjuvants",
        "analogs",
        "analogues",
        "homologues",
        "enzymes",
        "anions",
        "ions",
        "drugs",
        "regimens",
        "clones",
    ]

    term_sets = [
        ["modulators", "modulating", "modulations"],
        ["modifiers", "modifying", "modifications"],
        ["inhibitors", "inhibition", "inhibiting", "inhibits"],
        ["agonists", "agonizing", "agonizes", "agonisms"],
        ["antagonists", "antagonizing", "antagonizes", "antagonism"],
        [
            "activators",
            "activations",
            "activating",
        ],
        ["neuromodulations", "neuromodulators", "neuromodulating", "neuromodulates"],
        ["stimulators", "stimulations", "stimulating", "stimulates"],
        ["conjugations", "conjugating", "conjugates"],
        ["modulates", "modulates? binding"],
        ["(?:poly)peptides", "(?:poly)peptides? binding"],
        ["proteins", "proteins? binding"],
        ["(?:poly)?nucleotides", "(?:poly)?nucleotides? binding"],
        ["molecules", "molecules? binding"],
        ["ligands", "ligands? binding", "ligands? that bind"],
        ["fragments", "fragments? binding", "fragments? that bind"],
        ["promotion", "promoting", "promotes"],
        ["enhancements", "enhancing", "enhances", "enhancers"],
        ["regulators", "regulation", "regulating"],
    ]

    prefixes = [
        "cyclic",
        "reversible",
        "irreversible",
        "new",
        "bispecific",
        "bi-specific",
        "monoclonal",
        "acceptable",
        "receptor",
        "encoding",
        "encoders",
        "positive allosteric",
        "small molecule",
        "potent",
        "inventive",
        "selective",
        "novel",
        "heterocyclic",
        "partial",
        "inverse",
        "dual",
        "single",
    ]

    prefix_re = "|".join([p + " " for p in prefixes])

    def get_query(term: Union[str, list[str]], field: TextField):
        if isinstance(term, list):
            term = "(?:" + "?|".join(term) + ")"
        else:
            term = term + "?"
        sql = f"""
            UPDATE {TABLE} a
            SET original_term=(REGEXP_EXTRACT({field}, '(?i)((?:{prefix_re})*{term} (?:of |for |the |that |to |comprising )+ (?:(?:the|a) )?.*?)(?:and|useful|for|,|$)'))
            FROM `fair-abbey-386416.patents.gpr_publications` p
            WHERE p.publication_number=a.publication_number
            AND REGEXP_CONTAINS(original_term, "^(?i)(?:{prefix_re})*{term}$")
            AND REGEXP_CONTAINS(p.{field}, '(?i).*{term} (?:of|for|the|that|to|comprising).*')
        """
        return sql

    def get_hyphen_query(term):
        sql = f"""
            UPDATE {TABLE} a
            SET original_term=(REGEXP_EXTRACT(title, '(?i)([A-Za-z0-9]+-{term}?)'))
            FROM `fair-abbey-386416.patents.gpr_publications` p
            WHERE p.publication_number=a.publication_number
            AND REGEXP_CONTAINS(original_term, '^(?i){term}?$')
            AND REGEXP_CONTAINS(p.title, '(?i).*[A-Za-z0-9]+-{term}?.*')
        """
        return sql

    for term in terms:
        sql = get_query(term, "title")
        execute_bg_query(sql)

    for term in [*terms, *[t for term_set in term_sets for t in term_set]]:
        sql = get_hyphen_query(term)
        execute_bg_query(sql)

    # loop over term sets, in which the original_term may be in another form than the title variant
    for term_set in term_sets:
        for field in TEXT_FIELDS:
            sql = get_query(term_set, field)
            execute_bg_query(sql)


WordPlace = Literal["leading", "trailing", "all"]


def clean_annotations():
    """
    Remove trailing junk and silly matches
    """

    def get_remove_word(word, place: WordPlace):
        if place == "trailing":
            return f"""update `{TABLE}` set original_term=(REGEXP_REPLACE(original_term, '(?i) {word}$', '')) where regexp_contains(original_term, '(?i).* {word}$')"""
        elif place == "leading":
            return f"""update `{TABLE}` set original_term=(REGEXP_REPLACE(original_term, '(?i)^{word} ', '')) where regexp_contains(original_term, '(?i)^{word} .*')"""
        else:
            return rf"""update `{TABLE}` set original_term=(REGEXP_REPLACE(original_term, '(?i)(?:^|$| ){word}(?:^|$| )', ' ')) where regexp_contains(original_term, '(?i)(?:^|$| ){word}(?:^|$| )')"""

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
    }
    queries = [
        *[get_remove_word(word, place) for word, place in removal_words.items()],
        f"delete from `{TABLE}` where original_term is null",
        f"update `{TABLE}` "
        + "set original_term=(REGEXP_REPLACE(original_term, '[ ]{2,}', ' ')) where regexp_contains(original_term, '[ ]{2,}')",
        f"update `{TABLE}` set original_term=(REGEXP_REPLACE(original_term, '^[ ]+', '')) where regexp_contains(original_term, '^[ ]+')",
        f"update `{TABLE}` set original_term=(REGEXP_REPLACE(original_term, '[)]', '')) where original_term like '%)' and original_term not like '%(%';",
        f"delete from `{TABLE}` where original_term='COMPOSITION' or original_term='therapeutical' or original_term='prognosticating' or original_term in ('wherein a', 'pharmaceutical compositions', 'compound I', 'wherein said compound', 'fragment of', 'pharmacological compositions', 'therapeutically', 'general formula (I)', 'medicine Compounds', 'receptacle', 'liver', 'treatment regimens', 'unsubstituted', 'Compound I', 'medicinal compositions', 'COMPOUND', 'DISEASE', 'medicine Compounds of formula', 'THERAPY') or original_term like '% administration' or original_term like '% patients' or original_term like 'treat %' or original_term like 'treating %' or original_term like 'field of %'",
        f"update `{TABLE}` set domain='mechanisms' where original_term in ('tumor infiltrating lymphocytes')",
        f"update `{TABLE}` set original_term=(REGEXP_REPLACE(original_term, 'disease factor', 'disease')) where original_term like '% disease factor';",
        f"update `{TABLE}` set "
        + "original_term=regexp_extract(original_term, '(.{10,})(?:\\. [A-Z]\\w{3,}).*') where regexp_contains(original_term, '.{10,}\\. [A-Z]\\w{3,}')",
        f"delete FROM `{TABLE}` where length(original_term) < 2",
    ]
    for sql in queries:
        results = execute_bg_query(sql)


def fix_unmatched():
    """
    Example: 3 -d]pyrimidine derivatives -> Pyrrolo [2, 3 -d]pyrimidine derivatives
    """

    def get_query(field, char_set):
        sql = rf"""
            UPDATE {TABLE} a
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


if __name__ == "__main__":
    if "-h" in sys.argv:
        print("Usage: python3 ner.py\nCleans up annotations after NER is complete")
        sys.exit()

    # fix_of_for_annotations()

    # derivatives -> compounds
    # clean_annotations()
    # remove_substrings()

    fix_unmatched()
