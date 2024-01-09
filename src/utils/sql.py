from typing import Sequence

from typings import QueryType


def get_term_sql_query(terms: Sequence[str], query_type: QueryType = "AND") -> str:
    """
    Form a query from terms

    e.g. ["btk inhibitor", "aurora kinase inhibitor"] with query_type OR -> "(btk & inhibitor) | (aurora & kinase & inhibitor)"
    """
    # AND words within a given term
    # e.g. ["btk inhibitor", "aurora kinase inhibitor"] -> ["(btk & inhibitor)", "(aurora & kinase & inhibitor)"]
    anded_words = [f"({' & '.join(t.lower().split(' '))})" for t in terms]

    operand = "&" if query_type == "AND" else "|"
    term_query = f" {operand} ".join(anded_words)
    return f"({term_query})"
