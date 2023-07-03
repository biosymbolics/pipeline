"""
SynonymStore is a wrapper around redisearch to store synonyms and their metadata.
"""
from functools import reduce
import json
import logging
import os
import re
from typing import Any, Mapping, Optional, TypedDict
import uuid
from redis.exceptions import ResponseError  # type: ignore
import redisearch
from redisearch import TextField, IndexDefinition, Query

from common.utils.string import byte_dict_to_string_dict, get_id

RedisHashSet = Mapping[bytes, bytes]

REDIS_HOST = "redis-12973.c1.us-west-2-2.ec2.cloud.redislabs.com"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SynonymDocument(TypedDict):
    term: str
    canonical_id: str
    metadata: dict[str, str]


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
        self.client = redisearch.Client(
            index_name,
            port=12973,
            host=REDIS_HOST,
            password=os.environ.get("REDIS_KEY"),
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

    def __get_doc_id(self, term: str) -> str:
        return f"term:{get_id(term)}"

    def __deserialize(self, hashset: RedisHashSet) -> SynonymDocument:
        """
        Deserialize a Redis hashset into a SynonymDocument

        Args:
            hashset (RedisHashSet): The Redis hashset to deserialize
        """
        try:
            string_hs = byte_dict_to_string_dict(hashset)
            syn_doc: SynonymDocument = {
                "term": string_hs.get("term") or "",
                "canonical_id": string_hs.get("canonical_id") or "",
                "metadata": json.loads(string_hs.get("metadata") or "{}"),
            }
            return syn_doc
        except Exception as e:
            logger.error(
                "Failed to deserialize hashset %s (%s), %s",
                hashset,
                str(type(hashset)),
                e,
            )
            raise

    def __serialize(self, doc: SynonymDocument) -> RedisHashSet:
        """
        Serialize a SynonymDocument into a Redis hashset

        Args:
            doc (SynonymDocument): The SynonymDocument to serialize
        """
        return {
            b"term": bytes(self.__escape(doc["term"]), "utf-8"),
            b"canonical_id": bytes(doc["canonical_id"], "utf-8"),
            b"metadata": bytes(json.dumps(doc["metadata"]), "utf-8"),
        }

    def __escape(self, term: str) -> str:
        """
        Escape a term for RedisSearch
        TODO: more characters to escape?
        """
        replacements = {
            "^-": "",
            "-$": "",
            "-": "\\-",
            r"\(": "",
            r"\)": "",
            r"\[": "",
            r"\]": "",
            ",": "",
            r"\.": "\\.",
        }
        clean_term = reduce(
            lambda t, kv: re.sub(kv[0], kv[1], t), replacements.items(), term
        )
        return clean_term

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

    def __upsert_synonym(
        self,
        term: str,
        canonical_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SynonymDocument:
        """
        Upsert a synonym (add if new, otherwise update)
        The public interface is "add_synonym", which maps to an existing record if sufficiently similar.

        Args:
            term (str): The synonym to add
            canonical_id (str, optional): The canonical id of the synonym. Defaults to None.
            metadata (dict[str, Any], optional): Metadata to store with the synonym. Defaults to None.
        """
        doc_id = self.__get_doc_id(term)
        doc: SynonymDocument = {
            "term": term,
            "canonical_id": canonical_id or "",
            "metadata": metadata or {},
        }
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
        Add a synonym to the store and map it to the most similar term

        If canonical_id is provided, no mapping; the synonym is just added as a new record.

        Args:
            term (str): the synonym to add
            canonical_id (str, optional): the canonical id of the synonym. Defaults to None.
            metadata (dict[str, Any], optional): Metadata to store with the synonym. Defaults to None.
            distance (int): the maximum edit distance to search for
        """
        # only get docs if no canonical_id
        docs = self.search_for_synonyms(term, distance) if not canonical_id else []

        if not canonical_id and len(docs) > 0:
            most_similar_term = docs[0]["term"]
            new_canonical_id = docs[0]["canonical_id"]
            new_metadata = docs[0]["metadata"] or {"canonical_name": most_similar_term}
        else:
            new_canonical_id = canonical_id or self.__get_tmp_canonical_id()
            new_metadata = metadata or {}

        logger.info(
            "Added %s as synonym of %s (%s)", term, new_canonical_id, new_metadata
        )

        if not new_canonical_id or len(new_canonical_id.strip()) < 2:
            logging.error(
                "Canonical ID must be at least 2 characters long: %s, %s, %s",
                new_canonical_id,
                canonical_id,
                len(docs),
            )
            raise Exception("Canonical ID must be at least 2 characters long")

        upserted = self.__upsert_synonym(term, new_canonical_id, new_metadata)

        return upserted

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
                raise Exception("No synonym found")
        except Exception as e:
            logger.error("Error getting synonym %s: %s", term, e)
            return None

        return self.__deserialize(doc)

    def search_for_synonyms(
        self, term: str, distance: int = 10
    ) -> list[SynonymDocument]:
        """
        Search for a synonym in the store.

        Args:
            term (str): the synonym to search for
            distance (int): the maximum edit distance to search for
        """
        query = self.__prepare_query(term)
        logger.debug("Query for synonym: %s", query)

        try:
            q = Query(query).with_scores().paging(0, distance)
            result = self.client.search(q)
            hashsets = [self.client.redis.hgetall(doc.id) for doc in result.docs]
            return [self.__deserialize(hset) for hset in hashsets]
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
                    syn_doc["term"], new_canonical_id, syn_doc["metadata"]
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
        results = self.search_for_synonyms(term, distance)
        terms: list[str] = [doc["term"] for doc in results]
        self.remap_synonyms(new_canonical_id, [term, *terms])
        logger.info("Remapped %s to %s", terms, new_canonical_id)
