"""
To run after NER is complete
"""
from functools import reduce
import re
import logging
from typing import Sequence
from spacy.tokens import Doc

from system import initialize

initialize()


from utils.re import get_or_re

from .constants import (
    EXPANSION_ENDING_DEPS,
    EXPANSION_ENDING_POS,
    EXPANSION_NUM_CUTOFF_TOKENS,
    EXPANSION_POS_OVERRIDE_TERMS,
    TARGET_PARENS,
)
from .types import WordPlace

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def remove_trailing_leading(
    terms: Sequence[str], removal_terms: dict[str, WordPlace]
) -> Sequence[str]:
    logger.info("Removing trailing/leading words")

    def get_leading_trailing_re(place: str) -> re.Pattern | None:
        WB = "(?:^|$|[;,.: ])"  # no dash wb
        all_words = [t[0] + "s?[ ]*" for t in removal_terms.items() if t[1] == place]

        if len(all_words) == 0:
            return None

        or_re = get_or_re(all_words, "+")
        if place == "trailing":
            final_re = rf"{WB}{or_re}$"
        elif place == "conditional_trailing":
            # e.g. to avoid "bio-affecting substances" -> "bio-affecting"
            leading_exceptions = ["a", "an", "ed", "the", "ing", "ion", "ic"]
            le_re = "".join([f"(?<!{e})" for e in leading_exceptions])  # NOT OR'd!
            final_re = rf"{le_re}{WB}{or_re}$"
        elif place == "leading":
            final_re = rf"^{or_re}{WB}"
        elif place == "conditional_all":
            final_re = rf"(?<!(?:the|ing)){WB}{or_re}{WB}"
        else:
            final_re = rf"{WB}{or_re}{WB}"

        return re.compile(final_re, re.IGNORECASE | re.MULTILINE)

    # without p=p, lambda will use the last value of p
    steps = [
        lambda s, p=p, r=r: re.sub(get_leading_trailing_re(p), r, s)  # type: ignore
        for p, r in {
            "trailing": "",
            "conditional_trailing": "",
            "leading": "",
            "all": " ",
            "conditional_all": " ",
        }.items()
        if get_leading_trailing_re(p) is not None
    ]

    clean_terms = [reduce(lambda s, f: f(s), steps, term) for term in terms]
    return clean_terms


def expand_parens_term(text: str, original_term: str) -> str | None:
    """
    Returns expanded term in cases like 'agonists' -> '(sstr4) agonists'
    TODO: typically is more like 'somatostatin receptor subtype 4 (sstr4) agonists'

    Args:
        text (str): Text to search / full text
        original_term (str): Original term
    """
    possible_parens_term = re.findall(
        f"{TARGET_PARENS} {original_term}", text, re.IGNORECASE
    )

    if len(possible_parens_term) == 0:
        return None

    return possible_parens_term[0]


def expand_term(original_term: str, text: str, text_doc: Doc) -> str | None:
    """
    Returns expanded term
    Looks until it finds the next dobj or other suitable ending dep.
    @see https://downloads.cs.stanford.edu/nlp/software/dependencies_manual.pdf

    TODO:
    -  CCONJ - this is sometimes valuable, e.g. inhibitor of X and Y
    -  cd23  (ROOT, NOUN) antagonists  (dobj, NOUN) for  (prep, ADP) the  (det, DET) treatment  (pobj, NOUN) of  (prep, ADP) neoplastic  (amod, ADJ) disorders (pobj, NOUN)

    Args:
        original_term (str): Original term
        text (str): Text to search / full text
        text_doc (Doc): Spacy doc (passed in for perf reasons)
    """
    s = re.search(rf"\b{re.escape(original_term)}\b", text_doc.text, re.IGNORECASE)

    # shouldn't happen
    if s is None:
        logger.error("No term text in expansion: %s, %s", original_term, text)
        return None

    char_to_tok_idx = {t.idx: t.i for t in text_doc}

    # starting index of the string (we're only looking forward)
    start_idx = char_to_tok_idx.get(s.start())
    appox_orig_len = len(original_term.split(" "))

    # check -1 in case of hyphenated term in text
    # TODO: [number average molecular weight]×[content mass ratio] (content)
    if start_idx is None:
        start_idx = char_to_tok_idx.get(s.start() - 1)

    if start_idx is None:
        logger.error("start_idx none for %s\n%s", original_term, text)
        return None
    doc = text_doc[start_idx:]

    # syntactic subtree around the term
    subtree = list([d for d in doc[0].subtree if d.i >= start_idx])
    deps = [t.dep_ for t in subtree]

    ending_idxs = [
        deps.index(dep)
        for dep in EXPANSION_ENDING_DEPS
        # to avoid PRON (pronoun)
        if dep in deps and subtree[deps.index(dep)].pos_ in EXPANSION_ENDING_POS
        # don't consider if expansion contains "OR" or "AND" (tho in future we will)
        and "cc" not in deps[appox_orig_len : deps.index(dep)]
        # eg "inhibiting the expression of XYZ"
        and subtree[deps.index(dep)].text.lower() not in EXPANSION_POS_OVERRIDE_TERMS
    ]
    next_ending_idx = min(ending_idxs) if len(ending_idxs) > 0 else -1
    if next_ending_idx > 0 and next_ending_idx <= EXPANSION_NUM_CUTOFF_TOKENS:
        expanded = subtree[0 : next_ending_idx + 1]
        expanded_term = "".join([t.text_with_ws for t in expanded]).strip()
        return expanded_term

    return None
