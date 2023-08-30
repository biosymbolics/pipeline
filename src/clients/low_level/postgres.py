"""
Low-level Postgres client
"""
import time
from typing_extensions import NotRequired
from typing import Any, Mapping, TypeGuard, TypeVar, TypedDict
import logging
import psycopg
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

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
        "is_gin": NotRequired[bool],
        "is_lower": NotRequired[bool],
        "is_tgrm": NotRequired[bool],
        "is_uniq": NotRequired[bool],
    },
)
IndexSql = TypedDict("IndexSql", {"sql": str})


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_index_sql(index_def: IndexCreateDef | IndexSql) -> TypeGuard[IndexSql]:
    return index_def.get("sql") is not None


def is_index_create_def(
    index_def: IndexCreateDef | IndexSql,
) -> TypeGuard[IndexCreateDef]:
    return index_def.get("sql") is None


MIN_CONNECTIONS = 2
MAX_CONNECTIONS = 20


class NoResults(Exception):
    pass


class PsqlClient:
    _instances: dict[tuple, Any] = {}

    def __init__(
        self,
        uri: str = DATABASE_URL,
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

    def __init__(self, uri: str = DATABASE_URL):
        self.client = PsqlClient.get_instance(uri=uri)

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
            res = self.execute_query(query, [])
            return res["data"][0]["exists"]
        except Exception as e:
            logger.error("Error checking table exists: %s", e)
            return False

    def handle_error(
        self, conn, e: Exception, is_rollback: bool = False, ignore_error: bool = False
    ) -> ExecuteResult:
        if ignore_error:
            logger.info("Acceptable error executing query: %s", e)
            return {"data": [], "columns": {}}
        elif isinstance(e, NoResults):
            logger.debug("No results executing query (not an error)")
        else:
            logger.error("Error executing query (%s). Rolling back? %s", e, is_rollback)
            if is_rollback:
                conn.rollback()
            self.client.put_conn(conn)
            raise e

        self.client.put_conn(conn)
        return {"data": [], "columns": {}}

    @overrides(DatabaseClient)
    def execute_query(
        self,
        query: str,
        values: list = [],
        ignore_error: bool = False,
    ) -> ExecuteResult:
        """
        Execute query

        Args:
            query (str): SQL query
            ignore_error (bool): if True, will not raise an error if the query fails
        """
        start = time.time()
        logger.info("Starting query: %s", query)

        conn = self.client.get_conn()
        with conn.cursor() as cursor:
            try:
                cursor.execute(query, values)  # type: ignore
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

            try:
                if cursor.rownumber is None:
                    raise NoResults("Query returned no rows")

                data = list(cursor.fetchall())
                columns = dict(
                    [
                        (desc.name, desc._type_display())
                        for desc in (cursor.description or [])
                    ]
                )

                print(cursor.description, columns)
                self.client.put_conn(conn)
                return {"data": data, "columns": columns}
            except Exception as e:
                return self.handle_error(conn, e, ignore_error=ignore_error)

    @overrides(DatabaseClient)
    def _insert(self, table_name: str, records: list[T]) -> ExecuteResult:
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

    def create_index(self, index_def: IndexCreateDef | IndexSql):
        """
        Add an index
        """
        self.execute_query("CREATE EXTENSION pg_trgm", [], ignore_error=True)
        try:
            if is_index_sql(index_def):
                logger.info("Creating index: %s", index_def["sql"])
                self.execute_query(index_def["sql"], [])
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
                self.execute_query(sql, [])

            else:
                raise Exception("Invalid index def")
        except psycopg.errors.DuplicateTable as e:
            logger.warning("Index already exists: %s", e)

    def create_indices(self, index_defs: list[IndexCreateDef | IndexSql]):
        """
        Add indices
        """
        for index_def in index_defs:
            self.create_index(index_def)

    @staticmethod
    def copy_between_db(
        source_db: str,
        source_sql: str,
        dest_db: str,
        dest_table_name: str,
        truncate_if_exists: bool = True,
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

        # pull records from source db
        results = source_client.execute_query(source_sql)
        records = [dict(row) for row in results["data"]]

        # recreate
        dest_client.create_table(
            dest_table_name,
            results["columns"],
            exists_ok=True,
            truncate_if_exists=truncate_if_exists,
        )

        # add records
        dest_client.insert_into_table(records, dest_table_name)
