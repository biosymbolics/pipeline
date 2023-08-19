"""
Low-level Postgres client
"""
from typing_extensions import NotRequired
from typing import Mapping, TypeGuard, TypeVar, TypedDict
import logging
from psycopg2 import pool
from psycopg2.extras import DictCursor, execute_values
from urllib.parse import urlparse

from clients.low_level.database import DatabaseClient, ExecuteResult
from constants.core import DATABASE_URL
from typings.core import is_string_list
from utils.classes import overrides

T = TypeVar("T", bound=Mapping)
IndexCreateDef = TypedDict(
    "IndexCreateDef",
    {
        "table": str,
        "column": str,
        "is_trgm": NotRequired[bool],
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


logger = logging.getLogger(__name__)


MIN_CONNECTIONS = 2
MAX_CONNECTIONS = 20


class NoResults(Exception):
    pass


class PsqlClient:
    def __init__(
        self,
        uri: str = DATABASE_URL,
    ):
        parsed = urlparse(uri)
        username = parsed.username or ""
        password = parsed.password or ""
        database = parsed.path[1:]
        host = parsed.hostname
        port = parsed.port
        auth_args = (
            {"user": username, "password": password}
            if username is not None or password is not None
            else {}
        )
        self.conn_pool = pool.SimpleConnectionPool(
            minconn=MIN_CONNECTIONS,
            maxconn=MAX_CONNECTIONS,
            database=database,
            host=host,
            port=port,
            cursor_factory=DictCursor,
            **auth_args,
        )

    def get_conn(self):
        return self.conn_pool.getconn()

    def put_conn(self, conn):
        self.conn_pool.putconn(conn)

    def close_all(self):
        self.conn_pool.closeall()


class PsqlDatabaseClient(DatabaseClient):
    """
    Usage:
    ```
    from clients.low_level.postgres import PsqlDatabaseClient
    psql = PsqlDatabaseClient("patents")
    ```
    """

    def __init__(self, uri: str = DATABASE_URL):
        self.client = PsqlClient(uri)

    @staticmethod
    @overrides(DatabaseClient)
    def get_table_id(table_name: str) -> str:
        return table_name

    @overrides(DatabaseClient)
    def is_table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists
        """
        query = f"""
            SELECT EXISTS (
                SELECT FROM
                    information_schema.tables
                WHERE  table_name = '{table_name}'
            );
        """
        try:
            res = self.execute_query(query)
            return res["data"][0][0]
        except Exception as e:
            logger.error("Error checking table exists: %s", e)
            return False

    def handle_error(
        self, conn, e: Exception, is_rollback: bool = False, ignore_error: bool = False
    ) -> ExecuteResult:
        if ignore_error:
            logging.info("Acceptable error executing query: %s", e)
            return {"data": [], "columns": []}
        elif isinstance(e, NoResults):
            logging.debug("No results executing query (not an error)")
        else:
            logging.error(
                "Error executing query (%s). Rolling back? %s", e, is_rollback
            )
            if is_rollback:
                conn.rollback()
            self.client.put_conn(conn)
            raise e

        self.client.put_conn(conn)
        return {"data": [], "columns": []}

    @overrides(DatabaseClient)
    def execute_query(self, query: str, ignore_error: bool = False) -> ExecuteResult:
        """
        Execute query

        Args:
            query (str): SQL query
            ignore_error (bool): if True, will not raise an error if the query fails
        """
        logging.info("Starting query: %s", query)

        conn = self.client.get_conn()
        with conn.cursor() as cursor:
            try:
                cursor.execute(query)
                conn.commit()
            except Exception as e:
                return self.handle_error(
                    conn, e, is_rollback=True, ignore_error=ignore_error
                )

            try:
                if cursor.rowcount < 1 or cursor.pgresult_ptr is None:
                    raise NoResults("Query returned no rows")

                data = list(cursor.fetchall())
                columns = [desc[0] for desc in cursor.description]
                self.client.put_conn(conn)
                return {"data": data, "columns": columns}
            except Exception as e:
                return self.handle_error(conn, e, ignore_error=ignore_error)

    @overrides(DatabaseClient)
    def _insert(self, table_name: str, records: list[T]) -> ExecuteResult:
        columns = [c for c in list(records[0].keys())]
        values = [tuple(record.values()) for record in records]
        insert_cols = ", ".join([f'"{c}"' for c in columns])
        query = f"INSERT INTO {table_name} ({insert_cols}) VALUES %s"
        values = [tuple(item[col] for col in columns) for item in records]

        conn = self.client.get_conn()
        with conn.cursor() as cursor:
            try:
                execute_values(cursor, query, values)
                conn.commit()
                self.client.put_conn(conn)
                return {"data": [], "columns": columns}
            except Exception as e:
                return self.handle_error(conn, e, is_rollback=True)

    @overrides(DatabaseClient)
    def _create(self, table_name: str, columns: list[str] | dict[str, str]):
        """
        Simple create table function, makes up schema based on column names

        Args:
            table_name (str): name of the table
            columns (list[str]): list of columns or schema
        """
        table_id = self.get_table_id(table_name)
        if is_string_list(columns):
            schema = [f"{c} TEXT" for c in columns]
        elif isinstance(columns, dict):
            schema = [f"{c} {t}" for c, t in columns.items()]
        else:
            raise Exception("Invalid columns")

        query = f"CREATE TABLE {table_id} ({(', ').join(schema)});"
        self.execute_query(query)

    def create_indices(self, index_defs: list[IndexCreateDef | IndexSql]):
        """
        Add indices
        """
        self.execute_query("CREATE EXTENSION pg_trgm", ignore_error=True)

        for index_def in index_defs:
            if is_index_sql(index_def):
                self.execute_query(index_def["sql"])
                continue

            if is_index_create_def(index_def):
                table = index_def["table"]
                column = index_def["column"]
                is_tgrm = index_def.get("is_tgrm", False)
                is_uniq = index_def.get("is_uniq", False)

                if is_tgrm:
                    sql = f"CREATE INDEX trgm_index_{table}_{column} ON {table} USING gin (lower({column}) gin_trgm_ops)"
                else:
                    sql = f"CREATE {'UNIQUE' if is_uniq else ''} INDEX index_{table}_{column} ON {table} ({column})"

                self.execute_query(sql)

            else:
                raise Exception("Invalid index def")
