"""
Test script entity index
"""
import logging
import sys

from core import EntityIndex, SourceDocIndex
from common.utils.misc import dict_to_named_tuple
from prompts import GET_SIMPLE_TRIPLE_PROMPT

logging.getLogger().setLevel(logging.INFO)


def main(entity_list: list[str]):
    """
    Main
    """
    for entity in entity_list:
        source = dict_to_named_tuple({"doc_source": "SEC", "doc_type": "10-K"})
        si = SourceDocIndex()
        ei = EntityIndex(entity)
        si.load()
        ei.load()
        prompt = f"Summarize what we know about {entity}."  # GET_SIMPLE_TRIPLE_PROMPT(entity)
        answer1 = ei.query(prompt, source)
        answer2 = si.query(prompt, source)
        print("ANSWER1", answer1)
        print("ANSWER2", answer2)


if __name__ == "__main__":
    if len(sys.argv) == 1 or "-h" in sys.argv:
        print("Usage: python3 entity_test.py entity1 [entity2 ...]")
        sys.exit()
    main(sys.argv[1:])
