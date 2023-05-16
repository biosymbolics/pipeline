"""
Utilities for NER (or the poor man's equiv)
"""
import re
from pydash import title_case

# TODO: redis?
# synonym: canonical_name
entity_map = {"Lorlatinib": "Lorbrena/Lorviqua"}

RE_PARENS_NAME = r"(.*?)(?:\(.+)\)*"


def __clean_name(name_str: str) -> str:
    trimmed = name_str.strip()
    title_cased = title_case(trimmed)
    return title_cased


def parse_entity_names(entity_str: str) -> list[str]:
    """
    Parse names from an entity (assumes parenthetical is another name)
    e.g. "Lorbrena/Lorviqua (lorlatinib)" -> ["Lorbrena/Lorviqua", Lorlatinib]
    TODO:
    - remove short junk like "(a)"
    """
    matches = re.findall(RE_PARENS_NAME, entity_str)
    print("MATCHES", matches)
    names = list(map(__clean_name, matches))
    return names


def __ner_map_get_or_put(synonym: str, primary_name: str) -> str:
    match = entity_map.get(synonym)
    if match:
        return match

    entity_map[synonym] = primary_name
    return primary_name


def normalize_entity_name(entity_str: str) -> str:
    """
    Normalize entity name
    - parse out name(s)
    - lookup in map
    """
    names = parse_entity_names(entity_str)
    primary_name = names[0]

    normalized = list(map(lambda name: __ner_map_get_or_put(name, primary_name), names))
    return normalized[0]
