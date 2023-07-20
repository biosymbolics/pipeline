"""
To run after NER is complete
"""
import sys
from clients.low_level.big_query import execute_bg_query


def fix_of_for_annotations():
    # Define your terms
    terms = [
        "fragments",
        "modulators",
        "inhibitors",
        "inhibition",
        "blockers",
        "antagonists",
        "agonists",
        "ligands",
        "reagents",
        "derivatives",
        "antigens",
        "compositions",
        "activators",
        "linkers",
        "isoforms",
        "conjugates",
        "stereoisomers",
        "polypeptides",
        "surfactant",
        "antibodies",
        "antibody",
        "stimulator",
    ]

    # Loop over your terms
    for term in terms:
        # Construct the SQL statement
        sql = f"""
        UPDATE `fair-abbey-386416.patents.biosym_annotations` a
        SET original_term=(REGEXP_EXTRACT(title, '(?i)((?:new |bispecific |monoclonal |acceptable |receptor |positive allosteric |small molecule |potent |inventive |selective |novel |heterocyclic |partial |inverse )?{term}? (?:of|for) (?:(?:the|a)\\s)?.*?)(?:and|useful|for|,|$)'))
        FROM `fair-abbey-386416.patents.gpr_publications` p
        WHERE p.publication_number=a.publication_number
        AND REGEXP_CONTAINS(original_term, "^(?i)(?:new |bispecific |monoclonal |acceptable |receptor |positive allosteric |small molecule |potent |inventive |selective |novel |heterocyclic |partial |inverse )?{term}?$")
        AND REGEXP_CONTAINS(p.title, '(?i).*{term}? (?:of|for).*')
        """
        execute_bg_query(sql)


def clean_annotations():
    queries = [
        "update `fair-abbey-386416.patents.biosym_annotations` set original_term=(REGEXP_REPLACE(original_term, 'such', '')) where original_term like '% such';",
        "update  `fair-abbey-386416.patents.biosym_annotations` set original_term=(REGEXP_REPLACE(original_term, '[)]', '')) where original_term like '%)' and original_term not like '%(%';",
        "update `fair-abbey-386416.patents.biosym_annotations` set original_term=(REGEXP_REPLACE(original_term, 'thereof', '')) where original_term like '% thereof';",
        "delete from `fair-abbey-386416.patents.biosym_annotations` where original_term='COMPOSITION' or original_term='therapeutical' or original_term='prognosticating' or original_term in ('unsubstituted', 'medicine Compounds of formula', 'THERAPY') or original_term like '% administration' or original_term like '% patients' or original_term like 'treat %'",
    ]
    for sql in queries:
        execute_bg_query(sql)


if __name__ == "__main__":
    if "-h" in sys.argv:
        print("Usage: python3 ner.py\nCleans up annotations after NER is complete")
        sys.exit()

    fix_of_for_annotations()
    clean_annotations()
