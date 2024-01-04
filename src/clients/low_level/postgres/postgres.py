"""
Low-level Postgres client
"""
import time
from typing import Any, Awaitable, Callable, Sequence, cast
import logging
import uuid
import psycopg
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

from clients.low_level.database import (
    DatabaseClient,
    ExecuteResult,
    M,
    TransformFunction,
)
from constants.core import BASE_DATABASE_URL, DATABASE_URL
from typings.core import is_string_list
from utils.classes import overrides, nonoverride

from .types import (
    IndexCreateDef,
    IndexSql,
    is_index_create_def,
    is_index_sql,
    NoResults,
)
from .utils import get_schema_from_cursor


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


MIN_CONNECTIONS = 2
MAX_CONNECTIONS = 20


class PsqlClient:
    _instances: dict[tuple, Any] = {}

    def __init__(
        self,
        uri: str = DATABASE_URL or "",
    ):
        logger.info(
            "Creating new psql connection pool (min: %s, max: %s)",
            MIN_CONNECTIONS,
            MAX_CONNECTIONS,
        )
        self.conn_pool = ConnectionPool(
            min_size=MIN_CONNECTIONS,
            max_size=MAX_CONNECTIONS,
            conninfo=uri,
            kwargs={"row_factory": dict_row},
        )

    def get_conn(self):
        conn = self.conn_pool.getconn()
        return conn

    def put_conn(self, conn):
        self.conn_pool.putconn(conn)

    def close_all(self):
        self.conn_pool.close()

    @classmethod
    def get_instance(cls, **kwargs) -> "PsqlClient":
        # Convert kwargs to a hashable type
        kwargs_tuple = tuple(sorted(kwargs.items()))
        if kwargs_tuple not in cls._instances:
            logger.info("Creating new instance of %s", cls)
            cls._instances[kwargs_tuple] = cls(**kwargs)
        else:
            logger.info("Using existing instance of %s", cls)
        return cls._instances[kwargs_tuple]


class PsqlDatabaseClient(DatabaseClient):
    """
    Usage:
    ```
    from clients.low_level.postgres import PsqlDatabaseClient
    psql = PsqlDatabaseClient("patents")
    ```
    """

    def __init__(self, uri_or_db: str = DATABASE_URL or ""):
        if not uri_or_db.startswith("postgres://"):
            logger.warning("Passed non-uri string; assuming db name. Expanding to uri.")
            uri = f"{BASE_DATABASE_URL}/{uri_or_db}"
        else:
            uri = uri_or_db
        self.client = PsqlClient.get_instance(uri=uri)

    @staticmethod
    @overrides(DatabaseClient)
    def get_table_id(table_name: str) -> str:
        return table_name

    @overrides(DatabaseClient)
    async def is_table_exists(self, table_name: str) -> bool:
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
            res = await self.execute_query(query, [])
            return res["data"][0]["exists"]
        except Exception as e:
            logger.error("Error checking table exists: %s", e)
            return False

    @nonoverride
    def handle_error(
        self,
        conn,
        e: Exception,
        is_rollback: bool = False,
        ignore_error: bool = False,
        query: str | None = "",
    ) -> ExecuteResult:
        if ignore_error:
            logger.info("Acceptable error executing query: %s", e)
            return {"data": [], "columns": {}}
        elif isinstance(e, NoResults):
            logger.debug("No results executing query (not an error)")
        else:
            logger.error(
                "Error executing query %s (%s). Rolling back? %s",
                e,
                query or "unspecified",
                is_rollback,
            )
            if is_rollback:
                conn.rollback()
            self.client.put_conn(conn)
            raise e

        self.client.put_conn(conn)
        return {"data": [], "columns": {}}

    @overrides(DatabaseClient)
    async def execute_query(
        self,
        query: str,
        params: Sequence = [],
        handle_result_batch: Callable[[list], Awaitable[None]] | None = None,
        batch_size: int = 10000000,
        ignore_error: bool = False,
    ) -> ExecuteResult:
        """
        Execute query

        Args:
            query (str): SQL query (select, update, insert, etc)
            params (Sequence): params for query, if any (default: [])
            handle_result_batch (Callable): function to handle results in batches (default: None)
            batch_size (int): batch size for handle_result_batch (default: 10000000; only used if handle_result_batch is not None)
            ignore_error (bool): ignore error if True (default: False)
        """
        start = time.time()
        logger.info("Starting query: %s", query)

        conn = self.client.get_conn()
        with conn.cursor() as cursor:
            try:
                cursor.execute(query, params)  # type: ignore # TODO: LiteralString
                conn.commit()
                logger.info("Row count for query: %s", cursor.rowcount)
            except Exception as e:
                return self.handle_error(
                    conn,
                    e,
                    is_rollback=True,
                    ignore_error=ignore_error,
                )

            execute_time = round(time.time() - start, 2)
            if execute_time > 100:
                logger.info("Query took %s minutes", round(execute_time / 60, 2))

            # fetch results if appropriate
            try:
                if cursor.rownumber is None:
                    raise NoResults("Query returned no rows")

                if handle_result_batch is not None:
                    while rows := cursor.fetchmany(batch_size):
                        await handle_result_batch(rows)

                    self.client.put_conn(conn)
                    return {"data": [], "columns": {}}

                rows = list(cursor.fetchmany(batch_size))
                columns = get_schema_from_cursor(cursor)
                self.client.put_conn(conn)

                return {"data": cast(Sequence[dict], rows), "columns": columns}
            except Exception as e:
                return self.handle_error(conn, e, ignore_error=ignore_error)

    @overrides(DatabaseClient)
    def _insert(self, table_name: str, records: list[M]) -> ExecuteResult:
        columns = [c for c in list(records[0].keys())]
        insert_cols = ", ".join([f'"{c}"' for c in columns])
        query = f"INSERT INTO {table_name} ({insert_cols}) VALUES ({(', ').join(['%s' for _ in range(len(columns)) ])})"
        values = [[item[col] for col in columns] for item in records]

        conn = self.client.get_conn()
        with conn.cursor() as cursor:
            try:
                cursor.executemany(query, values)  # type: ignore
                conn.commit()
                self.client.put_conn(conn)
                return {"data": [], "columns": {}}
            except Exception as e:
                return self.handle_error(conn, e, query=query, is_rollback=True)

    @overrides(DatabaseClient)
    async def _create(self, table_name: str, columns: list[str] | dict[str, str]):
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
        await self.execute_query(query)

    @nonoverride
    async def create_index(self, index_def: IndexCreateDef | IndexSql):
        """
        Add an index
        """
        await self.execute_query("CREATE EXTENSION pg_trgm", [], ignore_error=True)
        try:
            if is_index_sql(index_def):
                logger.info("Creating index: %s", index_def["sql"])
                await self.execute_query(index_def["sql"], [])
                return

            if is_index_create_def(index_def):
                table = index_def["table"]
                column = index_def["column"]
                is_tgrm = index_def.get("is_tgrm", False)
                is_uniq = index_def.get("is_uniq", False)
                is_gin = index_def.get("is_gin", False)
                is_lower = index_def.get("is_lower", is_tgrm)
                col = f"lower({column})" if is_lower else column

                if is_tgrm:
                    sql = f"CREATE INDEX trgm_index_{table}_{column} ON {table} USING gin ({col} gin_trgm_ops)"
                elif is_gin:
                    sql = f"CREATE INDEX gin_index_{table}_{column} ON {table} USING gin ({col})"
                else:
                    sql = f"CREATE {'UNIQUE' if is_uniq else ''} INDEX index_{table}_{column} ON {table} ({col})"

                logger.info("Creating index: %s", sql)
                await self.execute_query(sql, [])

            else:
                raise Exception("Invalid index def")
        except psycopg.errors.DuplicateTable as e:
            logger.warning("Index already exists: %s", e)

    @nonoverride
    async def create_indices(self, index_defs: list[IndexCreateDef | IndexSql]):
        """
        Add indices
        """
        for index_def in index_defs:
            await self.create_index(index_def)

    @nonoverride
    @staticmethod
    async def copy_between_db(
        source_db: str,
        source_sql: str,
        dest_db: str,
        dest_table_name: str,
        truncate_if_exists: bool = True,
        transform: TransformFunction | None = None,
        transform_schema: Callable[[dict], dict] | None = None,
    ):
        """
        Copy data from one psql db to another

        Args:
            source_db (str): source db name
            source_sql (str): source sql query
            dest_db (str): destination db name
            dest_table (str): destination table name
        """
        dest_client = PsqlDatabaseClient(dest_db)
        source_client = PsqlDatabaseClient(source_db)

        results = await source_client.execute_query(source_sql)
        records = results["data"]
        logger.info("Records in memory")

        # transform schema if method provided
        if transform_schema:
            logger.info("Transforming records")
            schema = transform_schema(results["columns"])
        else:
            schema = results["columns"]

        logger.info("Creating table/inserting records (%s)", len(records))
        # add records
        await dest_client.create_and_insert(
            dest_table_name,
            records,
            schema,
            truncate_if_exists=truncate_if_exists,
            transform=transform,
        )
