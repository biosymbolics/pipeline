"""
Test script entity index
"""
import logging

from core.indices.entity_index import EntityIndex
from common.utils.misc import dict_to_named_tuple

logging.getLogger().setLevel(logging.INFO)


def main():
    """
    Main
    """
    source = dict_to_named_tuple(
        {"company": "PFE", "doc_source": "SEC", "doc_type": "10-K"}
    )
    ei = EntityIndex("elranatamab")
    ei.load(source)
    answer = ei.query("Tell me all about this drug", source)
    print(answer)


if __name__ == "__main__":
    main()
