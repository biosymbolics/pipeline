# alter table applications alter column priority_date type date USING priority_date::date;
BQ_TYPE_OVERRIDES = {
    "character_offset_start": "INTEGER",
    "character_offset_end": "INTEGER",
    "priority_date": "DATE",
    "embeddings": "VECTOR(64)",  # Untested as of 11/22/23
}


def determine_data_type(value, field: str | None = None):
    if field and field in BQ_TYPE_OVERRIDES:
        return BQ_TYPE_OVERRIDES[field]

    if isinstance(value, int):
        return "INTEGER"
    elif isinstance(value, float):
        return "FLOAT"
    elif isinstance(value, bool):
        return "BOOLEAN"
    elif isinstance(value, list):
        if len(value) > 0:
            dt = determine_data_type(value[0])
            return f"{dt}[]"
        return "TEXT[]"
    else:  # default to TEXT for strings or other data types
        return "TEXT"
