"""
Low-level Postgres client
"""
from typing import Mapping, TypeVar
import logging
import psycopg2
from psycopg2 import pool
from psycopg2.extras import DictCursor, execute_values

from clients.low_level.database import DatabaseClient, ExecuteResult
from typings.core import is_string_list
from utils.classes import overrides

T = TypeVar("T", bound=Mapping)


logger = logging.getLogger(__name__)

MIN_CONNECTIONS = 2
MAX_CONNECTIONS = 20


class NoResults(Exception):
    pass


class PsqlClient:
    def __init__(
        self,
        database_name: str,
        host: str = "localhost",
        port: int = 5432,
        user: str | None = None,
        password: str | None = None,
    ):
        auth_args = (
            {
                "user": user or "",
                "password": password or "",
            }
            if user is not None or password is not None
            else {}
        )
        self.conn_pool = pool.SimpleConnectionPool(
            minconn=MIN_CONNECTIONS,
            maxconn=MAX_CONNECTIONS,
            database=database_name,
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

    def __init__(self, database_name: str = "patents"):
        self.client = PsqlClient(database_name)

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
        self, conn, e: Exception, is_rollback: bool = False
    ) -> ExecuteResult:
        if isinstance(e, NoResults):
            logging.debug("No results executing query")
        else:
            logging.error(
                "Error executing query (%s). Rolling back? %s", e, is_rollback
            )

        if is_rollback:
            conn.rollback()
        self.client.put_conn(conn)
        return {"data": [], "columns": []}

    @overrides(DatabaseClient)
    def execute_query(self, query: str) -> ExecuteResult:
        """
        Execute query

        Args:
            query (str): SQL query
        """
        logging.info("Starting query: %s", query)

        conn = self.client.get_conn()
        with conn.cursor() as cursor:
            try:
                cursor.execute(query)
                conn.commit()
            except Exception as e:
                return self.handle_error(conn, e, is_rollback=True)

            try:
                if cursor.rowcount < 1 or cursor.pgresult_ptr is None:
                    raise NoResults("Query returned no rows")

                data = list(cursor.fetchall())
                columns = [desc[0] for desc in cursor.description]
                self.client.put_conn(conn)
                return {"data": data, "columns": columns}
            except Exception as e:
                return self.handle_error(conn, e)

    @overrides(DatabaseClient)
    def _insert(self, table_name: str, records: list[T]) -> ExecuteResult:
        columns = list(records[0].keys())
        values = [tuple(record.values()) for record in records]
        query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES %s"
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

    def add_indices(self, index_sql: list[str]):
        """
        Add indices
        """
        self.execute_query("CREATE EXTENSION pg_trgm")

        for sql in index_sql:
            self.execute_query(sql)
