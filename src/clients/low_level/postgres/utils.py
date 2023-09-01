import psycopg


def get_schema_from_cursor(cursor: psycopg.Cursor):
    """
    Get column defintion (schema) from cursor

    Args:
        cursor (psycopg.Cursor): cursor
    """
    schema = dict(
        [(desc.name, desc._type_display()) for desc in (cursor.description or [])]
    )
    return schema
