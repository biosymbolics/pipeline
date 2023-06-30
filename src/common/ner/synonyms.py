"""
SynonymStore is a wrapper around redisearch to store synonyms and their metadata.
"""
import json
import logging
from typing import Any, Optional, TypedDict, cast
import uuid
from redis.exceptions import ResponseError
import redisearch
from redisearch import TextField, IndexDefinition, Query


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
        self.client = redisearch.Client(index_name)
        try:
            self.client.info()
        except ResponseError:
            self.client.create_index(
                [TextField("term"), TextField("canonical_id"), TextField("metadata")],
                definition=IndexDefinition(
                    prefix=["term:", "canonical_id:", "metadata:"]
                ),
            )

    def add_synonym(
        self,
        term: str,
        canonical_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        Add a synonym to the store

        Args:
            term (str): The synonym to add
            canonical_id (str, optional): The canonical id of the synonym. Defaults to None.
            metadata (dict[str, Any], optional): Metadata to store with the synonym. Defaults to None.
        """
        if metadata is None:
            metadata = {}
        metadata_str = json.dumps(metadata)
        doc_id = f"term:{term}"
        self.client.add_document(
            doc_id, term=term, canonical_id=canonical_id or "", metadata=metadata_str
        )

    def get_synonym(self, term: str) -> Optional[SynonymDocument]:
        """
        Get a synonym from the store

        Args:
            term (str): the synonym to fetch
        """
        doc_id = f"term:{term}"
        doc = self.client.load_document(doc_id)

        if not doc:
            return None

        dict_doc = doc.__dict__

        if dict_doc.get("metadata"):
            dict_doc["metadata"] = json.loads(dict_doc["metadata"])

        return cast(SynonymDocument, dict_doc)

    def search_synonym(self, term: str, distance: int):
        """
        Search for a synonym in the store

        Args:
            term (str): the synonym to search for
            distance (int): the maximum edit distance to search for
        """
        q = Query(term).with_scores().paging(0, distance)
        results = self.client.search(q)
        return results

    def add_and_map_synonym(
        self, term: str, distance: int, metadata: Optional[dict[str, Any]] = None
    ):
        """
        Add a synonym to the store and map it to the most similar term

        Args:
            term (str): the synonym to add
            distance (int): the maximum edit distance to search for
            metadata (dict[str, Any], optional): Metadata to store with the synonym. Defaults to None.
        """
        result = self.search_synonym(term, distance)
        if result.total > 0:
            most_similar_term = result.docs[0].term
            canonical_id = (self.get_synonym(most_similar_term) or {})["canonical_id"]
            self.add_synonym(term, canonical_id, metadata)
        else:
            self.add_synonym(term, "temp-" + str(uuid.uuid4()), metadata=metadata)

    def remap_synonyms(self, new_canonical_id: str, synonyms: list[str]):
        """
        Remap all synonyms to the new canonical id

        Args:
            new_canonical_id (str): the new canonical id to map to
            synonyms (list[str]): the synonyms to remap
        """
        for term in synonyms:
            synonym_data = self.get_synonym(term)
            if synonym_data is not None:
                synonym_data["canonical_id"] = new_canonical_id
                self.client.add_document(f"term:{term}", **synonym_data)

    def remap_synonyms_by_search(self, term: str, distance: int, new_canonical_id: str):
        """
        Remap all synonyms that are similar to the given term to the new canonical id

        Args:
            term (str): the term around which to remap synonyms
            distance (int): the maximum edit distance to search for
            new_canonical_id (str): the new canonical id to map to
        """
        result = self.search_synonym(term, distance)
        terms: list[str] = [doc.term for doc in result.docs]
        self.remap_synonyms(new_canonical_id, [term, *terms])
        logging.info("Remapped %s to %s", terms, new_canonical_id)
