"""
Utils for the NER pipeline
"""

from functools import partial, reduce
import logging
from pydash import group_by
import regex as re
from typing import Iterable, Sequence
from spacy.tokens import Doc, Span, Token
import polars as pl

from constants.patterns.iupac import is_iupac
from utils.re import (
    RE_STANDARD_FLAGS,
    get_or_re,
    ReCount,
    ALPHA_CHARS,
)

from .spacy import Spacy
from .types import DocEntity


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
    core_infix_res: Sequence[str],
    soe_re: str = SOE_RE,
    eoe_re: str = EOE_RE,
    count: ReCount = 2,
) -> str:
    """
    Returns a regex for an entity with a prefix and suffix

    Args:
        core_infix_res (Sequence[str]): list of regexes for infixes
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
    core_suffix_res: Sequence[str],
    soe_re: str = SOE_RE,
    eoe_re: str = EOE_RE,
    prefix_count: ReCount = 2,
) -> str:
    """
    Returns a regex for an entity with a prefix and suffix

    Args:
        core_suffix_res (Sequence[str]): list of regexes for suffixes
        soe_re (str): start-of-entity regex (default: SOE_RE)
        eoe_re (str): end-of-entity regex (default: EOE_RE)
        prefix_count (ReCount): number of alpha chars in prefix
    """
    return soe_re + ALPHA_CHARS(prefix_count) + get_or_re(core_suffix_res) + eoe_re


def lemmatize_tail(
    term: str | Doc | Span,
    lemma_tags: Sequence[str] | None = None,
) -> str:
    """
    Lemmatizes the tail of a term

    e.g.
    "heart attacks" -> "heart attack"
    but not
    "fast reacting phenotypes" -> "fast reacting phenotype"

    Args:
        term (str): term to lemmatize
        lemma_tags (Sequence[str]): list of spacy tags to lemmatize (default: None)
    """
    if isinstance(term, str):
        nlp = Spacy.get_instance(disable=["ner"])
        doc = nlp(term)  # turn into spacy doc (has lemma info)
    elif isinstance(term, Doc) or isinstance(term, Span):
        doc = term
    else:
        raise ValueError("term must be a str or spacy Doc, but is %s", type(term))

    def maybe_lemmatize(token: Token, i: int) -> str:
        is_last = i == len(doc) - 1

        if not is_last:
            return token.text_with_ws

        if lemma_tags is None or token.tag_ in lemma_tags:
            return token.lemma_
        return token.text_with_ws

    # include all tokens as-is except for the last
    tail_lemmatized = "".join(
        [maybe_lemmatize(token, i) for i, token in enumerate(doc)]
    ).strip()

    return tail_lemmatized


def lemmatize_tails(
    terms: Sequence[str],
    n_process: int = 1,
) -> Iterable[str]:
    """
    Lemmatizes the tails of a list of terms

    e.g.
    "heart attacks" -> "heart attack"

    """
    nlp = Spacy.get_instance(disable=["ner"])
    docs = nlp.pipe(terms, n_process=n_process)  # turn into spacy docs

    for doc in docs:
        yield lemmatize_tail(doc)


def depluralize_tails(
    terms: Sequence[str],
    n_process: int = 1,
) -> Iterable[str]:
    """
    Singularize last terms

    e.g. "heart attacks" -> "heart attack"

    Args:
        terms (Sequence[str]): terms to normalize
        n_process (int): number of processes to use for parallelization
    """
    nlp = Spacy.get_instance(disable=["ner"])
    docs = nlp.pipe(terms, n_process=n_process)  # turn into spacy docs

    plural_pos_tags = ["NNS", "NNPS"]

    for doc in docs:
        yield lemmatize_tail(doc, lemma_tags=plural_pos_tags)


def lemmatize_all(term: str | Doc) -> str:
    """
    Lemmatizes all words in a string
    """
    if isinstance(term, str):
        nlp = Spacy.get_instance(disable=["ner"])
        doc = nlp(term)  # turn into spacy doc (has lemma info)
    elif isinstance(term, Doc):
        doc = term
    else:
        raise ValueError("term must be a str or spacy Doc, but is %s", type(term))

    # include all tokens as-is except for the last
    lemmatized = "".join(
        [f"{token.lemma_}{token.whitespace_}" for token in doc]
    ).strip()

    return lemmatized


def rearrange_terms(
    terms: Sequence[str], base_patterns: Sequence[str], n_process: int = 1
) -> Iterable[str]:
    """
    Perform term rearragement normalization

    - By ADP (e.g. "inhibitor of the kinase" -> "kinase inhibitor")
    - By base (e.g. "inhibiting XYZ" -> "XYZ inhibiting") (which will be subsequently normalized to 'inhibitor')

    Args:
        terms (Sequence[str]): terms to normalize
        base_patterns (Sequence[str]): patterns to match against for base rearrangement
        n_process (int): number of processes to use for parallelization
    """
    steps = [
        lambda _terms: rearrange_terms_by_base(_terms, base_patterns),
        lambda _terms: rearrange_adp_terms(_terms, n_process=n_process),
    ]

    return reduce(lambda x, func: func(x), steps, terms)


def rearrange_terms_by_base(
    terms: Sequence[str], base_re_patterns: Sequence[str]
) -> Iterable[str]:
    """
    Rearranges entity names that follow a known pattern of disorder,
    i.e. leading "inhibitor XYZ" -> trailing "XYZ inhibitor" (currently only supports two word terms)

    Args:
        terms (Sequence[str]): terms to normalize
        base_re_patterns (Sequence[str]): patterns to match against for base rearrangement
    """
    base_re = get_or_re(base_re_patterns)

    def rearrange_term_by_base(term: str) -> str:
        # base pattern followed by any non-space text
        # e.g. "inhibitor XYZ" -> "XYZ inhibitor" but not "inhibitor of XYZ antibody" -> "XYZ inhibitor"
        # (the latter is handled by rearrange_adp_terms)
        if len(term.split(" ")) != 2:
            return term
        return re.sub(f"^({base_re}) (.+)$", r"\2 \1", term, flags=RE_STANDARD_FLAGS)

    for term in terms:
        yield rearrange_term_by_base(term)


def rearrange_adp_terms(terms: Sequence[str], n_process: int = 1) -> Iterable[str]:
    """
    Rearranges & normalizes entity names with 'of' in them, e.g.
    turning "inhibitors of the kinase" into "kinase inhibitors"
    """

    # key: desired adposition term, value: regex that is mapped to adposition
    # e.g. caused by -> of
    adp_map = {
        "of": get_or_re(
            [
                r"(?:of|to|for)(?: use in)?(?: (?:the|a|an))?\b",
                r"(?:caus|mediat|characteri[sz]|influenc|modulat|regulat|relat)ed (?:by|to)(?: the|to|a)?\b",  # diseases characterized by reduced tgfb signaling -> tgfb reduced diseases (TODO)
                r"involving\b",  # diseases involving tgfβ -> tgfβ diseases
                r"targeting\b",
                r"for(?: the)?\b",  # agonist for the mglur5 receptor -> mglur5 agonist
                r"resulting(?: from)?\b",
                r"(?:relevant(?: to)?) to\b",
            ]
        ),
        "with": r"associated with\b",
        "against": r"against\b",  # ADP??
    }

    def _rearrange(_terms: Sequence[str], adp_term: str, adp_ext: str) -> Iterable[str]:
        normalized = [
            re.sub(adp_ext, adp_term, t, flags=RE_STANDARD_FLAGS) for t in _terms
        ]
        final = __rearrange_adp(normalized, adp_term=adp_term, n_process=n_process)
        return final

    steps = [
        partial(_rearrange, adp_term=term, adp_ext=ext) for term, ext in adp_map.items()
    ]

    text = reduce(lambda x, func: func(x), steps, terms)  # type: ignore
    return text


def __rearrange_adp(
    terms: Sequence[str], adp_term: str = "of", n_process: int = 1
) -> Iterable[str]:
    """
    Rearranges & normalizes entity names with 'of' in them, e.g.
    turning "inhibitors of the kinase" into "kinase inhibitors".

    Will recursively process multiple instances of the adposition, e.g.
    turning "conditions caused by up-regulation of IL-10" -> "IL-10 up-regulation conditions".

    ADP == adposition (e.g. "of", "with", etc.) (https://universaldependencies.org/u/pos/all.html#al-u-pos/ADP)
    """
    nlp = Spacy.get_instance(disable=["ner"])
    docs = nlp.pipe(terms, n_process=n_process)

    def __rearrange(doc: Doc | Span) -> str:
        # get indices of all ADPs, to use those as stopping points
        # refusing to consider the first ADP (if any) because it's likely to be a preposition
        all_adp_indices = [t.i for i, t in enumerate(doc) if t.pos_ == "ADP" and i > 0]

        # get ADP index for the specific term we're looking for
        specific_adp_indices = [
            t.i
            for i, t in enumerate(doc)
            if t.text == adp_term and t.pos_ == "ADP" and i > 0
        ]

        if len(specific_adp_indices) > 1:
            # recursively process doc
            head_doc = __rearrange(doc[specific_adp_indices[0] + 1 :].as_doc())
            tail_doc = __rearrange(doc[0 : specific_adp_indices[0]].as_doc())

            return f"{head_doc} {tail_doc}".strip()

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
            before_phrase = doc[before_start_idx:adp_index].text.strip()

            # all ADPs after
            after_adp_indices = all_adp_indices[rel_adp_index + 1 :]
            # stop at the first ADP after (if any)
            after_start_idx = (
                min(after_adp_indices) if len(after_adp_indices) > 0 else len(doc)
            )
            after_phrase = lemmatize_tail(doc[adp_index + 1 : after_start_idx])

            # if we cut off stuff from the beginning, put back
            # e.g. (diseases associated with) expression of GU Protein
            other_stuff = doc[0:before_start_idx].text if before_start_idx > 0 else ""

            return f"{other_stuff} {after_phrase} {before_phrase}".strip()
        except Exception as e:
            logger.error("Error in rearrange_adp, returning orig text: %s", e)
            return doc.text

    for doc in docs:
        yield __rearrange(doc)


def __normalize_by_pos(doc: Doc) -> str:
    """
    Normalizes a spacy doc by removing tokens based on their POS
    """
    logger.debug("Pos norm parts: %s", [(t.text, t.pos_) for t in doc])

    def clean_by_pos(t: Token, prev_t: Token | None, next_t: Token | None) -> str:
        # spacy only marks a token as SPACE if it is hanging out in a weird place
        if t.pos_ == "SPACE":
            return ""
        if t.text in ["'s", "’s"] and t.pos_ == "PART":
            # alzheimer's disease -> alzheimer disease
            return " "
        if t.text == DASH:
            if t.pos_ == "ADJ":
                return " "
            if t.pos_ in ["PUNCT", "SYM"]:
                if prev_t is None:
                    return " "
                if prev_t.pos_ == "PUNCT":
                    # "(-)-ditoluoyltartaric acid" unchanged
                    return DASH
                if (
                    next_t is not None
                    and (
                        (next_t.pos_ == "NUM" or len(next_t.text) < 3)
                        and prev_t.pos_ != "ADJ"  # avoid anti-pd1 -> antipd1
                    )
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


def normalize_by_pos(terms: Sequence[str]) -> Iterable[str]:
    """
    Normalizes entity by POS

    Dashes:
        Remove and replace with "" if Spacy considers PUNCT and followed by NUM:
            - APoE-4 -> apoe4 (NOUN(PUNCT)NUM)
            - HIV-1 -> hiv1 (NOUN(PUNCT)NUM)

        Remove and replace with **space** if Spacy considers it PUNCT, ADJ or SYM:
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
        - (-)-ditoluoyltartaric acid ((PUNCT)(PUNCT)(PUNCT)(PUNCT)NOUN) -> (-)-ditoluoyltartaric acid
        - Interleukin-9 for some reason is PROPN(SYM)NUM

        Keep if Spacy considers is a NOUN
        - HLA-C (NOUN(NOUN)NOUN) -> HLA-C # TODO??
        - IL-6 (NOUN(NOUN)NUM) -> IL-6 # TODO??

    Other changes:
        - Alzheimer's disease -> Alzheimer disease
    """
    nlp = Spacy.get_instance(disable=["ner"])

    # avoid spacy keeping terms with - as a single token
    def sep_dash(term: str) -> str:
        return (
            re.sub(DASHES_RE, rf" {DASH} ", term, flags=re.IGNORECASE)
            if not is_iupac(term)
            else term
        )

    sep_dash_terms = [sep_dash(term) for term in terms]
    docs = nlp.pipe(sep_dash_terms)

    for doc in docs:
        if is_iupac(doc.text):
            yield doc.text  # if iupac format, don't touch its parens
            continue

        yield __normalize_by_pos(doc)


def spans_to_doc_entities(spans: Iterable[Span]) -> list[DocEntity]:
    """
    Converts a list of spacy spans to a list of DocEntity

    Args:
        spans: list of spacy spans
    """
    entity_set = [
        DocEntity.create(
            span.text,
            span.start_char,
            span.end_char,
            normalized_term=span.text,  # just to init
            type=span.label_,
            vector=span.vector.tolist(),
            spacy_doc=span.as_doc(),
        )
        for span in spans
    ]
    return entity_set


def _create_cluster_term_map(
    terms: Sequence[str], cluster_ids: Sequence[int]
) -> dict[str, str]:
    """
    Creates a map between synonym/secondary terms and the primary
    For use with cluster_terms
    """
    term_clusters = (
        pl.DataFrame({"cluster_id": cluster_ids, "name": terms})
        .filter(pl.col("cluster_id") != -1)
        .group_by(["cluster_id", "name"])
        .len()
        .sort(["cluster_id", "name"])  # sort so that first name is the most common
        .group_by("cluster_id")  # re-cluster by cluster_id
        .agg(pl.col("name"))
        .drop("cluster_id")
        .to_series()
        .to_list()
    )
    return {
        m: members_terms[0]  # synonym to most-frequent-term
        for members_terms in term_clusters
        for m in members_terms
    }


def cluster_terms(terms: Sequence[str]) -> dict[str, str]:
    """
    Clusters terms using TF-IDF and DBSCAN.
    Returns a synonym record mapping between the canonical term and the synonym (one per pair)

    TODO:
    - returns one massive group (which it shouldn't because cluster -1 is the catch all)
    - try with BERT embeddings?

    Args:
        terms (Sequence[str]): list of terms to cluster
    """
    # lazy loading
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import DBSCAN

    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), strip_accents="unicode"
    )
    X = vectorizer.fit_transform(terms)
    clustering = DBSCAN(eps=0.6, min_samples=2).fit(X)  # HDBSCAN ?
    cluster_ids = list(clustering.labels_)

    return _create_cluster_term_map(terms, cluster_ids)
