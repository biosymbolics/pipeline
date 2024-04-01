import dataclasses, json
import decimal
from enum import Enum
from datetime import date

from typings.core import Dataclass, ResultBase


class DataclassJSONEncoder(json.JSONEncoder):
    """
    JSON encoder that supports dataclasses

    - casts dataclasses to dicts
    - casts dates to isoformat
    - casts enums to str

    TODO: move everything to pydantic
    """

    def default(self, o):
        if isinstance(o, Dataclass):
            return o.serialize()
        if isinstance(o, ResultBase):
            # pydantic base model for prisma entities
            return o.serialize()
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, date):
            return o.isoformat()
        if isinstance(o, Enum):
            return str(o)
        if isinstance(o, dataclasses.Field):
            return o.name
        if isinstance(o, decimal.Decimal):
            return float(o)
        return super().default(o)


class StorageDataclassJSONEncoder(DataclassJSONEncoder):
    """
    JSON encoder that supports dataclasses
    (for purposes of storage, which persists dataclasses without properties, expecting the result to be reconstituted with the constructor)
    """

    def default(self, o):
        if isinstance(o, Dataclass):
            return o.storage_serialize()
        return super().default(o)
