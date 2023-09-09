"""
SynonymStore is a wrapper around redisearch to store synonyms and their metadata.
"""
from functools import reduce
import json
import logging
import os
import regex as re
from typing import Any, Mapping, Optional, TypedDict, cast
import uuid
import Levenshtein
from redis.exceptions import ResponseError  # type: ignore
import redisearch
from redisearch import TextField, IndexDefinition, Query

from utils.string import byte_dict_to_string_dict, get_id

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

REDIS_HOST = "redis-12973.c1.us-west-2-2.ec2.cloud.redislabs.com"
REDIS_KEY = os.environ.get("REDIS_KEY")

RedisHashSet = Mapping[bytes, bytes]


class SynonymDocument(TypedDict):
    term: str
    canonical_id: str
    metadata: dict[str, str]


ESCAPE_MAP = {
    "^-": "",
    "-$": "",
    "-": r"\-",
    r"\(": "",  # TODO!! fix before rebuilding. should just escape.
    r"\)": "",
    r"\[": "",
    r"\]": "",
    ",": "",
    r"\.": r"\.",  # works because re.sub arg 2 is interpreted as string, not regex
}


class SynonymStore:
    """
    Wrapper around redisearch to store synonyms and their metadata.
    Attributes:
        client (redisearch.Client): RedisSearch client
    """

    def __init__(self, index_name: str = "synonyms"):
        """
        Initialize SynonymStore

        Args:
            index_name (str): The name of the index to use
        """
        if not REDIS_KEY:
            raise ValueError("REDIS_KEY not set")

        self.client = redisearch.Client(
            index_name,
            port=12973,
            host=REDIS_HOST,
            password=REDIS_KEY,
        )
        try:
            self.client.info()
            logger.info("Found index %s", index_name)
        except ResponseError:
            self.client.create_index(
                [TextField("term"), TextField("canonical_id"), TextField("metadata")],
                definition=IndexDefinition(),
            )
            logger.info("Created index %s", index_name)

    @staticmethod
    def __get_doc_id(term: str) -> str:
        return f"term:{get_id(term)}"

    @staticmethod
    def __deserialize(hashset: RedisHashSet) -> SynonymDocument:
        """
        Deserialize a Redis hashset into a SynonymDocument

        Args:
            hashset (RedisHashSet): The Redis hashset to deserialize
        """
        try:
            string_hs = byte_dict_to_string_dict(hashset)
            term = string_hs.get("term")
            syn_doc: SynonymDocument = {
                "term": SynonymStore.__unescape(term) if term else "",
                "canonical_id": string_hs.get("canonical_id") or "",
                "metadata": json.loads(string_hs.get("metadata") or "{}"),
            }
            return syn_doc
        except Exception as e:
            logger.error(
                "Failed to deserialize hashset %s, %s",
                hashset,
                e,
            )
            raise

    @staticmethod
    def __serialize(doc: SynonymDocument) -> RedisHashSet:
        """
        Serialize a SynonymDocument into a Redis hashset

        Args:
            doc (SynonymDocument): The SynonymDocument to serialize
        """
        return {
            b"term": bytes(SynonymStore.__escape(doc["term"]), "utf-8"),
            b"canonical_id": bytes(doc["canonical_id"], "utf-8"),
            b"metadata": bytes(json.dumps(doc["metadata"]), "utf-8"),
        }

    @staticmethod
    def __escape(term: str) -> str:
        """
        Escape a term for RedisSearch
        TODO: more characters to escape (https://stackoverflow.com/questions/65718424/redis-escape-special-character)
        """
        replacements = ESCAPE_MAP
        escaped_term = reduce(
            lambda t, kv: re.sub(kv[0], kv[1], t), replacements.items(), term
        )
        return escaped_term

    @staticmethod
    def __unescape(term: str) -> str:
        """
        Unescape a term from RedisSearch
        """
        replacements = {k: v for k, v in ESCAPE_MAP.items() if v != ""}
        # escapes for regex but also with extra \ (because ??)
        unescaped_term = reduce(
            lambda t, kv: re.sub(f"{kv[1]}|\\{kv[1]}", kv[0], t),
            replacements.items(),
            term,
        )
        return unescaped_term

    @staticmethod
    def __is_same(new_term: str, term_from_redis: str) -> bool:
        """
        Check if two terms are the same

        Args:
            new_term (str): The new term
            term_from_redis (str): The term from RedisSearch

        This will go away once escaping fixed to be reversible
        """
        return SynonymStore.__escape(new_term.lower()) == SynonymStore.__escape(
            term_from_redis.lower()
        )

    def __prepare_query(self, term: str) -> str:
        """
        Prepare a query for RedisSearch

        - Surround each subterm with a max Levenshtein distance of 3 (depends on term length)
        - Escape the term

        Args:
            term (str): The term to prepare

        Example:
            "PD1 inhibitors" ->  "@term:%PD1% %%%inhibitors%%%"
        """

        def __prep_subterm(subterm: str):
            lev_distance = min(3, round(len(subterm) / 3))
            return lev_distance * "%" + subterm + lev_distance * "%"

        subterms = [__prep_subterm(subterm) for subterm in term.split(" ")]

        query = self.__escape(" ".join(subterms))
        return "@term:" + query

    def __get_tmp_canonical_id(self) -> str:
        return "temp-" + str(uuid.uuid4())

    def __form_synonym_doc(
        self,
        term,
        canonical_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SynonymDocument:
        return {
            "term": term,
            "canonical_id": canonical_id or "",
            "metadata": metadata or {},
        }

    def __upsert_synonym(
        self,
        doc: SynonymDocument,
    ) -> SynonymDocument:
        """
        Upsert a synonym (add if new, otherwise update)
        The public interface is "add_synonym", which maps to an existing record if sufficiently similar.

        Args:
            doc (SynonymDocument): The synonym doc to upsert
        """
        term, canonical_id, metadata = doc["term"], doc["canonical_id"], doc["metadata"]
        logger.info("Upserting %s as synonym of %s (%s)", term, canonical_id, metadata)
        doc_id = SynonymStore.__get_doc_id(term)
        try:
            self.client.redis.hset(doc_id, mapping=self.__serialize(doc))  # type: ignore
            return doc
        except Exception as e:
            logger.error("Error upserting synonym %s but returning doc anyway", e)
            return doc

    def add_synonym(
        self,
        term: str,
        canonical_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        distance: int = 5,
    ) -> SynonymDocument:
        """
        (Maybe) add a synonym to the store and map it to the most similar term

        If canonical_id is provided, no mapping; the synonym is just added as a new record.

        Args:
            term (str): the synonym to add
            canonical_id (str, optional): the canonical id of the synonym. Defaults to None.
            metadata (dict[str, Any], optional): Metadata to store with the synonym. Defaults to None.
            distance (int): the maximum edit distance to search for
        """
        has_kg_match = canonical_id is not None

        docs = self.search(term, distance, True)
        most_similar_term = docs[0]["term"] if len(docs) > 0 else None
        found_self = (
            self.__is_same(term, most_similar_term) if most_similar_term else False
        )

        if found_self:
            logger.info("Found record %s; no updates", term)
            # nothing more to do; return (to avoid doing anything else resource-intensive)
            return docs[0]

        if has_kg_match:
            """
            UMLS match; not found in store
            """
            doc = self.__form_synonym_doc(term, canonical_id, metadata)
        elif not has_kg_match and len(docs) > 0:
            """
            No UMLS match but a non-self search result
            """
            new_metadata = docs[0]["metadata"] or {
                "canonical_name": most_similar_term or ""
            }
            doc = self.__form_synonym_doc(term, docs[0]["canonical_id"], new_metadata)
        else:
            # entirely new
            doc = self.__form_synonym_doc(term, self.__get_tmp_canonical_id(), metadata)

        self.__upsert_synonym(doc)
        return doc

    def get_synonym(self, term: str) -> Optional[SynonymDocument]:
        """
        Get a synonym from the store

        Args:
            term (str): the synonym to fetch
        """
        doc_id = self.__get_doc_id(term)

        try:
            doc = self.client.redis.hgetall(doc_id)

            if not doc:
                raise LookupError("No synonym found")
        except LookupError:
            return None
        except Exception as e:
            logger.error("Error getting synonym %s: %s", term, e)
            return None

        return self.__deserialize(doc)

    def search(self, *args, **kwargs) -> list[SynonymDocument]:
        """
        Alias for search_for_synonyms
        """
        return self.search_for_synonyms(*args, **kwargs)

    def search_for_synonyms(
        self, term: str, distance: int = 10, prefer_exact: bool = True
    ) -> list[SynonymDocument]:
        """
        Search for a synonym in the store.

        Args:
            term (str): the synonym to search for
            distance (int): the maximum edit distance to search for
            prefer_exact (bool): whether to return an exact match first
        """
        # fast lane for exact matches
        if prefer_exact:
            exact = self.get_synonym(term)
            if exact:
                return [exact]

        query = self.__prepare_query(term)
        logger.debug("Query for synonym: %s", query)

        def is_similar(doc: SynonymDocument) -> bool:
            return Levenshtein.distance(doc["term"], term) < distance

        try:
            q = Query(query).with_scores().paging(0, distance)
            result = self.client.search(q)
            hashsets = [self.client.redis.hgetall(doc.id) for doc in result.docs]
            docs = [self.__deserialize(hset) for hset in hashsets]
            return [doc for doc in docs if is_similar(doc)]
        except Exception as e:
            logger.error("Error searching for synonym %s: %s", term, e)
            return []

    def remap_synonyms(self, new_canonical_id: str, synonyms: list[str]):
        """
        Remap all synonyms to the new canonical id

        Args:
            new_canonical_id (str): the new canonical id to map to
            synonyms (list[str]): the synonyms to remap
        """
        for synonym in synonyms:
            syn_doc = self.get_synonym(synonym)
            if syn_doc is not None:
                self.__upsert_synonym(
                    cast(SynonymDocument, {**syn_doc, "canonical_id": new_canonical_id})
                )
                logger.info("Remapped %s to %s", syn_doc["term"], new_canonical_id)
            else:
                logger.info("Could not find synonym %s", synonym)

    def remap_synonyms_by_search(self, term: str, distance: int, new_canonical_id: str):
        """
        Remap all synonyms that are similar to the given term to the new canonical id

        Args:
            term (str): the term around which to remap synonyms
            distance (int): the maximum edit distance to search for
            new_canonical_id (str): the new canonical id to map to
        """
        results = self.search(term, distance)
        terms: list[str] = [doc["term"] for doc in results]
        self.remap_synonyms(new_canonical_id, [term, *terms])
        logger.info("Remapped %s to %s", terms, new_canonical_id)
