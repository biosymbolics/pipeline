from typing import Union, List, Dict

JsonSerializable = Union[
    Dict[str, "JsonSerializable"], List["JsonSerializable"], str, int, float, bool, None
]