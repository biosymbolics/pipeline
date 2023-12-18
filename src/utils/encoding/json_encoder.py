import dataclasses, json
import decimal
from enum import Enum
from datetime import date

from typings.core import Dataclass


class DataclassJSONEncoder(json.JSONEncoder):
    """
    JSON encoder that supports dataclasses

    - casts dataclasses to dicts
    - casts dates to isoformat
    - casts enums to str
    """

    def default(self, o):
        if isinstance(o, Dataclass):
            return o.asdict()
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
