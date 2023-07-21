"""
To run after NER is complete
"""
import sys
import logging

from system import initialize

initialize()

from clients.low_level.big_query import execute_bg_query

TABLE = "fair-abbey-386416.patents.biosym_annotations"


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
        "enzymes",
        "anions",
        "ions",
        "drugs",
    ]

    term_sets = [
        ["modulators", "modulating", "modulations"],
        ["modifiers", "modifying", "modifications"],
        ["inhibitors", "inhibition", "inhibiting", "inhibits"],
        ["agonists", "agonizing", "agonizes", "agonism"],
        ["antagonists", "antagonizing", "antagonizes", "antagonism"],
        [
            "activators",
            "activations",
            "activating",
        ],
        ["neuromodulations", "neuromodulators", "neuromodulating", "neuromodulates"],
        ["simulators", "simulations", "simulating", "simulates"],
        ["conjugations", "conjugating", "conjugates"],
        ["modulates", "modulates? binding"],
        ["(?:poly)peptides", "(?:poly)peptides? binding"],
        ["proteins", "proteins? binding"],
        ["(?:poly)?nucleotides", "(?:poly)?nucleotides? binding"],
        ["molecules", "molecules? binding"],
        ["ligands", "ligands? binding", "ligands? that bind"],
        ["fragments", "fragments? binding", "fragments? that bind"],
        ["promotion", "promoting", "promotes"],
        ["enhancement", "enhancing", "enhances", "enhancer"],
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

    def get_query(term, second_term=None):
        if not second_term:
            second_term = term
        sql = f"""
            UPDATE {TABLE} a
            SET original_term=(REGEXP_EXTRACT(title, '(?i)((?:{prefix_re})*{term}? (?:of|for|the|that|to) (?:(?:the|a) )?.*?)(?:and|useful|for|,|$)'))
            FROM `fair-abbey-386416.patents.gpr_publications` p
            WHERE p.publication_number=a.publication_number
            AND REGEXP_CONTAINS(original_term, "^(?i)(?:{prefix_re})*{second_term}?$")
            AND REGEXP_CONTAINS(p.title, '(?i).*{term}? (?:of|for).*')
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
        sql = get_query(term)
        execute_bg_query(sql)

    # for term in [*terms, *[t for term_set in term_sets for t in term_set]]:
    #     sql = get_hyphen_query(term)
    #     execute_bg_query(sql)

    # loop over term sets, in which the original_term may be in another form than the title variant
    for term_set in term_sets:
        for term in term_set:
            sql = get_query(term, term_set[0])
            execute_bg_query(sql)


def clean_annotations():
    """
    Remove trailing junk and silly matches
    """
    queries = [
        f"update `{TABLE}` set original_term=(REGEXP_REPLACE(original_term, ' such', '')) where original_term like '% such';",
        f"update `{TABLE}` set original_term=(REGEXP_REPLACE(original_term, 'such ', '')) where original_term like 'such %';",
        f"update `{TABLE}` set original_term=(REGEXP_REPLACE(original_term, 'the ', '')) where original_term like 'the %';",
        f"update `{TABLE}` set original_term=(REGEXP_REPLACE(original_term, 'excellent', '')) where original_term like '%excellent%';",
        f"update `{TABLE}` set original_term=(REGEXP_REPLACE(original_term, 'particular ', '')) where original_term like 'particular %';",
        f"update `{TABLE}` set original_term=(REGEXP_REPLACE(original_term, ' useful', '')) where original_term like '% useful';",
        f"update  `{TABLE}` set original_term=(REGEXP_REPLACE(original_term, '[)]', '')) where original_term like '%)' and original_term not like '%(%';",
        f"update `{TABLE}` set original_term=(REGEXP_REPLACE(original_term, 'thereof', '')) where original_term like '% thereof';",
        f"delete from `{TABLE}` where original_term='COMPOSITION' or original_term='therapeutical' or original_term='prognosticating' or original_term in ('wherein said compound', 'fragment of', 'therapeutically', 'general formula (I)', 'medicine Compounds', 'liver', 'treatment regimens', 'unsubstituted', 'Compound I', 'medicinal compositions', 'COMPOUND', 'DISEASE', 'medicine Compounds of formula', 'THERAPY') or original_term like '% administration' or original_term like '% patients' or original_term like 'treat %' or original_term like 'treating %' or original_term like 'field of %'",
        f"delete from `{TABLE}` where original_term is null",
        f"update `{TABLE}` set domain='mechanisms' where original_term in ('tumor infiltrating lymphocytes')",
    ]
    for sql in queries:
        results = execute_bg_query(sql)


if __name__ == "__main__":
    if "-h" in sys.argv:
        print("Usage: python3 ner.py\nCleans up annotations after NER is complete")
        sys.exit()

    fix_of_for_annotations()
    clean_annotations()
