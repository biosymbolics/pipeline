from typing import Sequence, TypeGuard

from typings.umls import UmlsRecord


def is_umls_record_list(
    records: Sequence[dict],
) -> TypeGuard[Sequence[UmlsRecord]]:
    """
    Check if list of records is a list of UmlsRecords
    """
    return (
        isinstance(records, list)
        and len(records) > 0
        and isinstance(records[0], dict)
        and "id" in records[0]
        and "canonical_name" in records[0]
        and "hierarchy" in records[0]
        and "type_id" in records[0]
        and "type_name" in records[0]
        and "level" in records[0]
    )
