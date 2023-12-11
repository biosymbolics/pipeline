from typing import Mapping, Sequence
from pydash import flatten, uniq
from spacy.tokens import Doc

from constants.patterns.iupac import is_iupac
from core.ner.types import CanonicalEntity, DocEntity
from data.domain.biomedical.umls import clean_umls_name
from utils.string import generate_ngram_phrases

from .candidate_selector import CandidateSelector

NGRAMS_N = 2
MIN_WORD_LENGTH = 1


class CompositeCandidateSelector(CandidateSelector):
    """
    A candidate generator that if not finding a suitable candidate, returns a composite candidate

    Look up in UMLS here and use 'type' for standard ordering on composite candidate names (e.g. gene first)
    select  s.term, array_agg(type_name), array_agg(type_id), ids from (select term, regexp_split_to_array(id, '\\|') ids from terms) s, umls_lookup, unnest(s.ids) as idd  where idd=umls_lookup.id and array_length(ids, 1) > 1 group by s.term, ids;

    - Certain gene names are matched naively (e.g. "cell" -> CEL gene, tho that one in particular is suppressed)

    TODO:
        pde-v inhibitor  - works of pde-v but not pde v or pdev
        bace 2 inhibitor - base2
        glp-2 agonist - works with dash
        'at1 receptor antagonist'
        "hyperproliferative disease cancer"
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.select_candidate = super().select_candidate

    @staticmethod
    def _is_composite_eligible(entity: DocEntity) -> bool:
        """
        Is a text a composite candidate?

        - False if it's an IUPAC name, which are just too easy to mangle (e.g. 3'-azido-2',3'-dideoxyuridine matching 'C little e')
        - false if it's too short (a single token or word)
        - Otherwise true
        """
        tokens = entity.spacy_doc or entity.term.split(" ")
        if is_iupac(entity.term):
            return False
        if len(tokens) < MIN_WORD_LENGTH:
            return False
        return True

    @classmethod
    def _get_ngrams(cls, entity: DocEntity, n: int) -> list[tuple[str, list[float]]]:
        """
        Get all ngrams in a text
        """
        if entity.spacy_doc is None:
            raise ValueError("Entity must have a spacy_doc")

        # only non-punct tokens
        tokens = [t for t in entity.spacy_doc if t.pos_ != "PUNCT"]
        doc = entity.spacy_doc

        # if fewer words than n, just return words
        # (this is expedient but probably confusing)
        if n == 1 or len(tokens) < n:
            return [
                (token.text, list(doc[i : i + 1].vector))
                for i, token in enumerate(tokens)
            ]

        ngrams = generate_ngram_phrases(doc, n)
        return ngrams

    def _form_composite_name(self, member_candidates: Sequence[CanonicalEntity]) -> str:
        """
        Form a composite name from the candidates from which it is comprised
        """

        def get_name_part(c: CanonicalEntity):
            if c.id in self.kb.cui_to_entity:
                ce = self.kb.cui_to_entity[c.id]
                return clean_umls_name(
                    ce.concept_id, ce.canonical_name, ce.aliases, ce.types, True
                )
            return c.name

        name = " ".join([get_name_part(c) for c in member_candidates])
        return name

    def _form_composite(
        self, members: Sequence[CanonicalEntity]
    ) -> CanonicalEntity | None:
        """
        Form a composite from a list of member entities
        """

        if len(members) == 0:
            return None

        # if just a single composite match, treat it like a non-composite match
        if len(members) == 1:
            return members[0]

        # sorted for the sake of consist composite ids
        ids = sorted([c.id for c in members])

        # form name from comprising candidates
        name = self._form_composite_name(members)

        return CanonicalEntity(
            id="|".join(ids),
            ids=ids,
            name=name,
            # description=..., # TODO: composite description
            # aliases=... # TODO: all permutations
        )

    def _generate_composite(
        self,
        entity: DocEntity,
        ngram_entity_map: Mapping[str, CanonicalEntity],
    ) -> CanonicalEntity | None:
        """
        Generate a composite candidate from a mention text

        Args:
            mention_text (str): Mention text
            ngram_entity_map (dict[str, MentionCandidate]): word-to-candidate map
        """
        if entity.spacy_doc is None:
            raise ValueError("Entity must have a spacy_doc")

        if entity.term.strip() == "":
            return None

        def get_composite_candidates(tokens: Doc) -> list[CanonicalEntity]:
            """
            Recursive function to see if the first ngram has a match, then the first n-1, etc.
            """
            if len(tokens) == 0:
                return []

            if len(tokens) >= NGRAMS_N:
                ngram = tokens[0:NGRAMS_N].text
                if ngram in ngram_entity_map:
                    remaining_words = tokens[NGRAMS_N:].as_doc()
                    return [
                        ngram_entity_map[ngram],
                        *get_composite_candidates(remaining_words),
                    ]

            # otherwise, let's map only the first word
            remaining_words = tokens[1:].as_doc()
            if tokens[0] in ngram_entity_map:
                return [
                    ngram_entity_map[tokens[0].text],
                    *get_composite_candidates(remaining_words),
                ]

            # otherwise, no match. create a fake MentionCandidate.
            return [
                # concept_id is the word itself, so
                # composite id will look like "UNMATCHED|C1999216" for "UNMATCHED inhibitor"
                CanonicalEntity(
                    name=tokens[0].text.lower(),
                    id=tokens[0].text.lower(),
                ),
                *get_composite_candidates(remaining_words),
            ]

        candidates = get_composite_candidates(entity.spacy_doc)

        return self._form_composite(candidates)

    # def _optimize_composites(
    #     self, composite_matches: dict[str, CanonicalEntity | None]
    # ) -> dict[str, CanonicalEntity | None]:
    #     """
    #     Taking the new composite names, see if there is now a singular match
    #     (e.g. a composite name might be "SGLT2 inhibitor", comprised of two candidates, for which a single match exists)
    #     """
    #     composite_names = uniq(
    #         [cm.name for cm in composite_matches.values() if cm is not None]
    #     )
    #     direct_match_map = {
    #         n: self._get_best_canonical(c)
    #         for n, c in zip(composite_names, self._get_candidates(composite_names))
    #     }
    #     # combine composite and potential single matches
    #     return {
    #         t: (direct_match_map.get(cm.name) if cm is not None else cm) or cm
    #         for t, cm in composite_matches.items()
    #     }

    def _generate_composite_entities(
        self, matchless_entity: DocEntity
    ) -> CanonicalEntity | None:
        """
        For a list of mention text without a sufficiently similar direct match,
        generate a composite match from the individual words

        Args:
            matchless_entity (DocEntity): a doc entity (NER span)
        """

        # create 1 and 2grams
        matchless_ngrams = uniq(
            flatten(
                [self._get_ngrams(matchless_entity, i + 1) for i in range(NGRAMS_N)]
            )
        )

        # get candidates for all ngrams
        ngram_entities = [self.select_candidate(*ngram) for ngram in matchless_ngrams]
        ngram_map = {
            ngram[0]: e
            for ngram, e in zip(matchless_ngrams, ngram_entities)
            if e is not None
        }

        # generate the composites
        composite_match = self._generate_composite(matchless_entity, ngram_map)

        # optimized_matches = self._optimize_composites(composite_matches)

        return composite_match

    def __call__(self, entity: DocEntity) -> CanonicalEntity | None:
        """
        Generate candidates for a list of mention texts

        If the initial top candidate isn't of sufficient similarity, generate a composite candidate.
        """
        candidates = self._get_candidates(entity.term)

        match = self._get_best_canonical(candidates, entity.embeddings)

        if match is not None:
            return match

        if CompositeCandidateSelector._is_composite_eligible(entity):
            return self._generate_composite_entities(entity)

        return match
