from typing import TypeGuard, TypedDict
from typing_extensions import NotRequired

IndexCreateDef = TypedDict(
    "IndexCreateDef",
    {
        "table": str,
        "column": str,
        "is_gin": NotRequired[bool],
        "is_lower": NotRequired[bool],
        "is_tgrm": NotRequired[bool],
        "is_uniq": NotRequired[bool],
    },
)
IndexSql = TypedDict("IndexSql", {"sql": str})


def is_index_sql(index_def: IndexCreateDef | IndexSql) -> TypeGuard[IndexSql]:
    return index_def.get("sql") is not None


def is_index_create_def(
    index_def: IndexCreateDef | IndexSql,
) -> TypeGuard[IndexCreateDef]:
    return index_def.get("sql") is None


class NoResults(Exception):
    pass
