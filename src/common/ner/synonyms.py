"""
SynonymStore is a wrapper around redisearch to store synonyms and their metadata.
"""
import json
import logging
import os
from typing import Any, Mapping, Optional, TypedDict, cast
import uuid
from redis.exceptions import ResponseError  # type: ignore
import redisearch
from redisearch import TextField, IndexDefinition, Query

from common.utils.string import get_id

RedisHashSet = Mapping[bytes, bytes]


class SynonymDocument(TypedDict):
    term: str
    canonical_id: str
    metadata: dict[str, str]


class SynonymStore:
    """
    Wrapper around redisearch to store synonyms and their metadata.
    """

    def __init__(self, index_name: str):
        """
        Initialize SynonymStore

        Args:
            index_name (str): The name of the index to use
        """
        self.client = redisearch.Client(
            index_name,
            port=12973,
            host="redis-12973.c1.us-west-2-2.ec2.cloud.redislabs.com",
            password=os.environ.get("REDIS_KEY"),
        )
        try:
            self.client.info()
            logging.info("Found index %s", index_name)
        except ResponseError:
            self.client.create_index(
                [TextField("term"), TextField("canonical_id"), TextField("metadata")],
                definition=IndexDefinition(
                    prefix=["term:", "canonical_id:", "metadata:"]
                ),
            )
            logging.info("Created index %s", index_name)

    def __get_doc_id(self, term: str) -> str:
        return f"term:{get_id(term)}"

    def __deserialize(self, hashset: RedisHashSet) -> SynonymDocument:
        syn_doc: SynonymDocument = {
            "term": str(hashset.get(b"term")) or "",
            "canonical_id": str(hashset.get(b"canonical_id")) or "",
            "metadata": json.loads(hashset[b"metadata"])
            if hashset.get(b"metadata")
            else {},
        }
        return syn_doc

    def __serialize(self, doc: SynonymDocument) -> RedisHashSet:
        return {
            b"term": bytes(doc["term"], "utf-8"),
            b"canonical_id": bytes(doc["canonical_id"], "utf-8"),
            b"metadata": bytes(json.dumps(doc["metadata"]), "utf-8"),
        }

    def add_new_synonym(
        self,
        term: str,
        canonical_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        Add a new synonym to the store

        Args:
            term (str): The synonym to add
            canonical_id (str, optional): The canonical id of the synonym. Defaults to None.
            metadata (dict[str, Any], optional): Metadata to store with the synonym. Defaults to None.
        """
        doc_id = self.__get_doc_id(term)
        doc: Mapping = {
            "term": term,
            "canonical_id": canonical_id or "",
            "metadata": metadata or {},
        }
        self.client.redis.hset(doc_id, mapping=doc)

    def add_synonym(
        self,
        term: str,
        canonical_id: Optional[str] = None,
        distance: int = 5,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        Add a synonym to the store and map it to the most similar term

        Args:
            term (str): the synonym to add
            distance (int): the maximum edit distance to search for
            metadata (dict[str, Any], optional): Metadata to store with the synonym. Defaults to None.
        """
        docs = self.search_synonym(term, distance)
        if len(docs) > 0:
            most_similar_term = docs[0].term
            canonical_id = docs[0].canonical_id
            self.add_new_synonym(term, canonical_id, metadata)
            logging.info(
                "Added %s as synonym of %s (%s)", term, most_similar_term, canonical_id
            )
        else:
            self.add_new_synonym(term, "temp-" + str(uuid.uuid4()), metadata=metadata)

    def get_synonym(self, term: str) -> Optional[SynonymDocument]:
        """
        Get a synonym from the store

        Args:
            term (str): the synonym to fetch
        """
        doc_id = self.__get_doc_id(term)
        doc = self.client.redis.hgetall(doc_id)

        if not doc:
            return None

        return self.__deserialize(doc)

    def search_synonym(self, term: str, distance: int = 10):
        """
        Search for a synonym in the store,

        Args:
            term (str): the synonym to search for
            distance (int): the maximum edit distance to search for
        """
        q = Query(term).with_scores().paging(0, distance)
        result = self.client.search(q)
        return result.docs

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
                self.add_new_synonym(
                    syn_doc["term"], new_canonical_id, syn_doc["metadata"]
                )  # TODO: remove existing doc?
                logging.info("Remapped %s to %s", syn_doc["term"], new_canonical_id)
            else:
                logging.info("Could not find synonym %s", synonym)

    def remap_synonyms_by_search(self, term: str, distance: int, new_canonical_id: str):
        """
        Remap all synonyms that are similar to the given term to the new canonical id

        Args:
            term (str): the term around which to remap synonyms
            distance (int): the maximum edit distance to search for
            new_canonical_id (str): the new canonical id to map to
        """
        results = self.search_synonym(term, distance)
        terms: list[str] = [doc.term for doc in results]
        self.remap_synonyms(new_canonical_id, [term, *terms])
        logging.info("Remapped %s to %s", terms, new_canonical_id)
