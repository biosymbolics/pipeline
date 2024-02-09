from dataclasses import asdict, dataclass, replace
from typing import (
    Any,
    TypeGuard,
    Union,
    List,
    Dict,
)
import inspect
from pydantic import BaseModel
from spacy.tokens import Doc

JsonSerializable = Union[
    Dict[str, "JsonSerializable"], List["JsonSerializable"], str, int, float, bool, None
]

Primitive = bool | str | int | float | None


@dataclass(frozen=True)
class Dataclass:
    """
    TODO: get rid of this in favor of pydantic
    """

    def __getitem__(self, item):
        # e.g. e[0], *e[0:6]
        if isinstance(item, int) or isinstance(item, slice):
            return list(self.values())[item]
        return getattr(self, item)

    def get(self, item, default=None):
        return getattr(self, item, default)

    def asdict(self):
        o = asdict(self)
        return o

    def _asdict(self) -> dict[str, Any]:
        return self.asdict()

    def to_dict(self) -> dict[str, Any]:
        return self.asdict()

    def keys(self):
        return self.asdict().keys()

    def values(self):
        return self.asdict().values()

    def items(self):
        return self.asdict().items()

    def replace(self, **kwargs):
        return replace(self, **kwargs)

    def serialize(self) -> dict[str, Any]:
        """
        Prep for JSON serialization of the object, including properties
        """
        properties = {
            key: getattr(self, key)
            for key, value in inspect.getmembers(self.__class__)
            if isinstance(value, property)
        }

        o = self.asdict()
        return {**o, **properties}

    def storage_serialize(self) -> dict[str, Any]:
        """
        Prep for JSON serialization of the object, WITHOUT properties (ie expects to be reconstituted into dataclass)
        """
        return self.asdict()


class ResultBase(BaseModel):
    """
    Base class to add to Prisma entities
    """

    def serialize(self) -> dict[str, Any]:
        """
        Prep for JSON serialization of the object, including properties

        """
        return self.model_dump()


def is_string_list(obj: Any) -> TypeGuard[list[str]]:
    """
    Checks if an object is a list of strings

    Args:
        obj (Any): object to check

    Returns:
        bool: True if the object is a list of strings
    """
    return isinstance(obj, list) and all(isinstance(x, str) for x in obj)
