"""
Low-level Postgres client
"""
from typing import Mapping, TypeVar
import logging
import psycopg2
import psycopg2.extras

from clients.low_level.database import DatabaseClient
from typings.core import is_string_list
from utils.classes import overrides

T = TypeVar("T", bound=Mapping)

logger = logging.getLogger(__name__)


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
        conn = psycopg2.connect(
            database=database_name,
            host=host,
            port=port,
            cursor_factory=psycopg2.extras.DictCursor,
            *auth_args,
        )
        self.conn = conn

    def cursor(self):
        return self.conn.cursor()


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
            return res[1][0][0]
        except Exception as e:
            logger.error("Error checking table exists: %s", e)
            return False

    @overrides(DatabaseClient)
    def execute_query(self, query: str):
        """
        Execute query

        Args:
            query (str): SQL query
        """
        logging.info("Starting query: %s", query)

        with self.client.cursor() as cursor:
            cursor.execute(query)

            try:
                data = list(cursor.fetchall())
                columns = [desc[0] for desc in cursor.description]
            except Exception as e:
                logging.error("Error fetching data: %s", e)
                return [], []

        return data  # columns

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
        return self.execute_query(query)
