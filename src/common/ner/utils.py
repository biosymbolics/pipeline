"""
Utils for the NER pipeline
"""
from functools import partial, reduce
import logging
import re
import time
from typing import Iterable
from spacy.tokens import Doc
from clients.spacy import Spacy

from common.utils.re import (
    get_or_re,
    ReCount,
    ALPHA_CHARS,
)

# end-of-entity regex
EOE_RE = "\\b" + ".*"

# start-of-entity regex
SOE_RE = ".*"

DASH = "-"
DASHES = ["–", "-"]
DASHES_RE = rf"[{''.join(DASHES)}]"


def get_entity_re(
    core_entity_re: str,
    soe_re: str = SOE_RE,
    eoe_re: str = EOE_RE,
    is_case_insensitive=False,
) -> str:
    """
    Returns a regex for an entity with a start-of-entity (soe_re) and end-of-entity (eoe_re) regexes

    Args:
        core_entity_re (str): regex for the core entity
        soe_re (str): start-of-entity regex (default: SOE_RE)
        eoe_re (str): end-of-entity regex (default: EOE_RE)
        is_case_insensitive (bool): whether to make the regex case insensitive (default: False)
    """
    core_re = soe_re + core_entity_re + eoe_re

    if is_case_insensitive:
        return "(?i)" + core_re

    return core_re


def get_infix_entity_re(
    core_infix_res: list[str],
    soe_re: str = SOE_RE,
    eoe_re: str = EOE_RE,
    count: ReCount = 2,
) -> str:
    """
    Returns a regex for an entity with a prefix and suffix

    Args:
        core_infix_res (list[str]): list of regexes for infixes
        soe_re (str): start-of-entity regex (default: SOE_RE)
        eoe_re (str): end-of-entity regex (default: EOE_RE)
        count (ReCount): number of alpha chars in prefix and suffix
    """
    return (
        soe_re
        + ALPHA_CHARS(count)
        + get_or_re(core_infix_res)
        + ALPHA_CHARS(count)
        + eoe_re
    )


def get_suffix_entitiy_re(
    core_suffix_res: list[str],
    soe_re: str = SOE_RE,
    eoe_re: str = EOE_RE,
    prefix_count: ReCount = 2,
) -> str:
    """
    Returns a regex for an entity with a prefix and suffix

    Args:
        core_suffix_res (list[str]): list of regexes for suffixes
        soe_re (str): start-of-entity regex (default: SOE_RE)
        eoe_re (str): end-of-entity regex (default: EOE_RE)
        prefix_count (ReCount): number of alpha chars in prefix
    """
    return soe_re + ALPHA_CHARS(prefix_count) + get_or_re(core_suffix_res) + eoe_re


def lemmatize_tail(term: str | Doc) -> str:
    """
    Lemmatizes the tail of a term

    e.g.
    "heart attacks" -> "heart attack"
    but not
    "fast reacting phenotypes" -> "fast reacting phenotype"
    """
    if isinstance(term, str):
        nlp = Spacy.get_instance()
        doc = nlp(term)  # turn into spacy doc (has lemma info)
    elif isinstance(term, Doc):
        doc = term
    else:
        raise ValueError("term must be a str or spacy Doc, but is %s", type(term))

    # include all tokens as-is except for the last
    tail_lemmatized = "".join(
        [
            token.text_with_ws if i < len(doc) - 1 else token.lemma_
            for i, token in enumerate(doc)
        ]
    ).strip()

    return tail_lemmatized


def lemmatize_tails(terms: list[str], n_process: int = 1) -> Iterable[str]:
    """
    Lemmatizes the tails of a list of terms
    """
    nlp = Spacy.get_instance()._nlp
    docs = nlp.pipe(terms, n_process=n_process)  # turn into spacy docs

    for doc in docs:
        yield lemmatize_tail(doc)


def rearrange_terms(terms: list[str], n_process: int = 1) -> Iterable[str]:
    """
    Rearranges & normalizes entity names with 'of' in them, e.g.
    turning "inhibitors of the kinase" into "kinase inhibitors"

    ADP == adposition (e.g. "of", "with", etc.) (https://universaldependencies.org/u/pos/all.html#al-u-pos/ADP)
    """
    adp_map = {
        "of": r"of (?:the|a|an)\b",
        "with": r"associated with\b",
        # "by": r"mediated by\b",
    }

    def _rearrange(_terms: list[str], adp_term: str, adp_ext: str) -> Iterable[str]:
        subbed = [re.sub(adp_ext, adp_term, t) for t in _terms]
        final = __rearrange_adp(subbed, adp_term=adp_term, n_process=n_process)
        return final

    steps = [
        partial(_rearrange, adp_term=term, adp_ext=ext) for term, ext in adp_map.items()
    ]

    text = reduce(lambda x, func: func(x), steps, terms)  # type: ignore
    return text


def __rearrange_adp(
    terms: list[str], adp_term: str = "of", n_process: int = 1
) -> Iterable[str]:
    """
    Rearranges & normalizes entity names with 'of' in them, e.g.
    turning "inhibitors of the kinase" into "kinase inhibitors"

    ADP == adposition (e.g. "of", "with", etc.) (https://universaldependencies.org/u/pos/all.html#al-u-pos/ADP)
    """
    nlp = Spacy.get_instance()._nlp
    docs = nlp.pipe(terms, n_process=n_process)

    def __rearrange(doc: Doc) -> str:
        tokens = doc
        # get indices of all ADPs, to use those as stopping points
        all_adp_indices = [t.i for t in tokens if t.pos_ == "ADP"]

        # get ADP index for the specific term we're looking for
        specific_adp_indices = [
            t.i for t in tokens if t.text == adp_term and t.pos_ == "ADP"
        ]
        adp_index = specific_adp_indices[0] if len(specific_adp_indices) > 0 else None

        if adp_index is None:
            return doc.text

        # relative adp index (== index on all_adp_list)
        rel_adp_index = all_adp_indices.index(adp_index)

        try:
            # all ADPs before the current one
            before_adp_indices = all_adp_indices[0:rel_adp_index]
            # start at the closest one before (if any)
            before_start_idx = (
                max(before_adp_indices) + 1 if len(before_adp_indices) > 0 else 0
            )
            before_tokens = tokens[before_start_idx:adp_index]
            before_phrase = "".join([t.text_with_ws for t in before_tokens]).strip()

            # all ADPs after
            after_adp_indices = all_adp_indices[rel_adp_index + 1 :]
            # stop at the first ADP after (if any)
            after_start_idx = (
                min(after_adp_indices) if len(after_adp_indices) > 0 else len(tokens)
            )
            after_tokens = tokens[adp_index + 1 : after_start_idx]
            after_phrase = lemmatize_tail(
                "".join([t.text_with_ws for t in after_tokens])
            )

            # if we cut off stuff from the beginning, put back
            # e.g. (diseases associated with) expression of GU Protein
            other_stuff = (
                "".join([t.text_with_ws for t in tokens[0:before_start_idx]])
                if before_start_idx > 0
                else ""
            )

            return f"{other_stuff}{after_phrase} {before_phrase}"
        except Exception as e:
            logging.error("Error in rearrange_adp, returning orig text: %s", e)
            return doc.text

    for doc in docs:
        yield __rearrange(doc)


def __normalize_by_pos(doc: Doc):
    """
    Normalizes a spacy doc by removing tokens based on their POS
    """

    def clean_by_pos(t, prev_t, next_t):
        # spacy only marks a token as SPACE if it is hanging out in a weird place
        if t.pos_ == "SPACE":
            return ""
        if t.text == "'s" and t.pos_ == "PART":
            # alzheimer's disease -> alzheimer disease
            return " "
        if t.text == DASH:
            if t.pos_ == "ADJ":
                return " "
            if t.pos_ == "PUNCT":
                if (
                    next_t is not None
                    and (next_t.pos_ == "NUM" or len(next_t.text) < 3)
                    and len(prev_t.text) < 5
                ):
                    # ApoE-4 -> apoe4
                    # IL-6 -> IL6
                    # Interleukin-6 -> Interleukin 6
                    return ""
                return " "
            else:
                # pos_ == NOUN, PROPN, etc
                return DASH

        if next_t is not None and next_t.text in DASHES:
            return t.text  # omitting pre-dash space

        return t.text_with_ws

    tokens = doc
    return "".join(
        [
            clean_by_pos(
                t,
                (tokens[i - 1] if i > 0 else None),
                (tokens[i + 1] if len(tokens) > (i + 1) else None),
            )
            for i, t in enumerate(tokens)
        ]
    )


def normalize_by_pos(terms: list[str], n_process: int = 1) -> Iterable[str]:
    """
    Normalizes entity by POS

    Dashes:
        Remove and replace with "" if Spacy considers PUNCT and followed by NUM:
            - APoE-4 -> apoe4 (NOUN(PUNCT)NUM)
            - HIV-1 -> hiv1 (NOUN(PUNCT)NUM)

        Remove and replace with **space** if Spacy considers it PUNCT or ADJ:
        - sodium channel-mediated diseases (NOUN(PUNCT)VERB)
        - neuronal hypo-kinetic disease (NOUN(PUNCT)ADJ) # TODO: better if ""
        - Loeys-Dietz syndrome (PROPN(PUNCT)NOUN)
        - sleep-wake cycles (NOUN(PUNCT)NOUN)
        - low-grade prostate cancer (ADJ(PUNCT)NOUN)
        - non-insulin dependent diabetes mellitus (ADJ(ADJ)NOUN)
        - T-cell lymphoblastic leukemia (NOUN(ADJ)NOUN)
        - T-cell (NOUN(PUNCT)NOUN)
        - MAGE-A3 gene (PROPN(PUNCT)NOUN) # TODO: better if ""
        - Bcr-Abl (NOUN(ADJ)ADJ) -> Bcr-Abl # TODO
        - HIV-1 (NOUN(PUNCT)PROPN) -> HIV-1 # TODO

        Keep if Spacy considers is a NOUN
        - HLA-C (NOUN(NOUN)NOUN) -> HLA-C # TODO??
        - IL-6 (NOUN(NOUN)NUM) -> IL-6 # TODO??

    Other changes:
        - Alzheimer's disease -> Alzheimer disease
    """
    nlp = Spacy.get_instance()

    def skip(term: str) -> bool:
        # simple hack to avoid futzing with things like '1-(3-aminophenyl)-6,8-dimethyl-5-...'
        return len(term.split(DASH)) > 3 and re.match(r"[0-9]+", term) is not None

    # avoid spacy keeping terms with - as a single token
    def sep_dash(term: str) -> str:
        return re.sub(DASHES_RE, rf" {DASH} ", term) if not skip(term) else term

    sep_dash_terms = [sep_dash(term) for term in terms]
    docs = nlp.pipe(sep_dash_terms, n_process=n_process)

    for doc in docs:
        if skip(doc.text):
            yield doc.text
            continue

        yield __normalize_by_pos(doc)
