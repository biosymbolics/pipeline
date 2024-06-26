from abc import abstractmethod
import logging
import time
from typing import Any, Callable, Mapping, Sequence, TypeVar, TypedDict
import logging
from typings.core import Dataclass

from utils.list import batch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

M = TypeVar("M", bound=Mapping | Dataclass)
ExecuteResult = TypedDict(
    "ExecuteResult", {"columns": dict[str, str], "data": Sequence[dict]}
)

# receives batch and all records, outputs a transformed batch
TransformFunction = Callable[[Sequence[M], Sequence[M]], Sequence[M]]


class DatabaseClient:
    def __init__(self):
        self.client = None  # Override

    @staticmethod
    @abstractmethod
    def get_table_id(table_name: str) -> str:
        raise NotImplementedError

    @abstractmethod
    async def execute_query(self, query: str, values: Sequence = []) -> ExecuteResult:
        raise NotImplementedError

    @abstractmethod
    async def is_table_exists(self, table_name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def _insert(
        self, table_name: str, records: Sequence[M], on_conflict: str | None = None
    ):
        raise NotImplementedError

    @abstractmethod
    async def _create(
        self,
        table_name: str,
        columns: Sequence[str] | Mapping[str, str],
    ):
        raise NotImplementedError

    async def create_from_select(self, query: str, new_table_name: str):
        """
        Create a new table from a query
        Drops table if existing.

        Args:
            query (str): SQL query
            new_table_name (str): name of the new table
        """
        logging.info("Creating table %s", new_table_name)
        new_table_id = self.get_table_id(new_table_name)
        create_table_query = f"""
            DROP TABLE IF EXISTS {new_table_id};
            CREATE TABLE {new_table_id} AS {query};
        """
        await self.execute_query(create_table_query)

    async def select(self, query: str, values: Sequence = []) -> list[dict]:
        """
        Execute a query and return the results as a list of dicts
        (must include provide fully qualified table name in query)

        Args:
            query (str): SQL query
        """
        logger.debug("Running query: %s (%s)", query, values)
        results = await self.execute_query(query, values)
        return [dict(row) for row in results["data"]]

    async def select_insert_into_table(self, select_query: str, table_name: str):
        """
        Insert rows into a table from a select query

        Args:
            select_query (str): select query
            table_name (str): name of the table
        """
        table_id = self.get_table_id(table_name)
        query = f"INSERT INTO {table_id} {select_query}"
        logger.info("Inserting via query (%s) into table %s", query, table_id)
        await self.execute_query(query)

    async def create_and_insert(
        self,
        table_name: str,
        records: Sequence[M],
        columns: Sequence[str] | Mapping[str, str] | None = None,
        truncate_if_exists: bool = True,
        transform: TransformFunction | None = None,
        batch_size: int = 1000,
    ):
        """
        Create a table and insert rows into it

        Args:
            records (Sequence[dict]): list of records to insert
            table_name (str): name of the table
            batch_size (int, optional): number of records to insert per batch. Defaults to 1000.
        """
        schema = columns or list(records[0].keys())
        await self.create_table(
            table_name, schema, exists_ok=True, truncate_if_exists=truncate_if_exists
        )
        await self.insert_into_table(records, table_name, transform, batch_size)

    async def insert_into_table(
        self,
        records: Sequence[M],
        table_name: str,
        transform: TransformFunction | None = None,
        batch_size: int = 1000,
        on_conflict: str | None = None,
    ):
        """
        Insert rows into a table from a list of records

        Args:
            records (Sequence[dict]): list of records to insert
            table_name (str): name of the table
            transform (TransformFunction, optional): function to transform records before inserting. Defaults to None.
            batch_size (int, optional): number of records to insert per batch. Defaults to 1000.
        """
        batched = batch(records, batch_size)

        for i, b in enumerate(batched):
            logging.info("Inserting batch %s into table %s", i, table_name)
            transformed = transform(b, records) if transform is not None else b
            await self._insert(table_name, transformed, on_conflict)

            logging.info("Successfully inserted %s rows", len(b))

    async def create_table(
        self,
        table_name: str,
        schema: Sequence[str] | Mapping[str, str],
        exists_ok: bool = True,
        truncate_if_exists: bool = False,
    ):
        """
        Create a table

        Args:
            table_name (str): name of the table (with or without dataset prefix)
            schema: list of column names or dict (column name -> type)
            exists_ok (bool): if True, do not raise an error if the table already exists
            truncate_if_exists (bool): if True, truncate the table if it already exists
        """
        logging.info("Creating table: %s", table_name)

        if truncate_if_exists and not exists_ok:
            raise Exception("Cannot truncate if exists if exists_ok is False")

        if await self.is_table_exists(table_name):
            if truncate_if_exists:
                await self.truncate_table(table_name)
            elif not exists_ok:
                raise Exception(f"Table {table_name} already exists")
            else:
                logging.info("Table %s already exists", table_name)
        else:
            await self._create(table_name, schema)

    async def delete_table(self, table_name: str, is_cascade: bool = False):
        """
        Delete a table (if exists)

        Args:
            table_name (str): name of the table
            is_cascade (bool): if True, delete all tables that depend on this table
        """
        table_id = self.get_table_id(table_name)
        logger.info("Deleting table %s", table_name)
        delete_table_query = (
            f"DROP TABLE IF EXISTS {table_id} {'CASCADE' if is_cascade else ''};"
        )
        await self.execute_query(delete_table_query)
        logging.info("Deleted table %s", table_name)

    async def truncate_table(self, table_name: str):
        """
        Truncate a table (if exists)

        Args:
            table_name (str): name of the table
        """
        exists = await self.is_table_exists(table_name)
        if not exists:
            logger.warning(
                "Table %s does not exist and thus not truncating", table_name
            )
            return

        table_id = self.get_table_id(table_name)
        logger.info("Truncating table %s", table_id)
        truncate_table_query = f"TRUNCATE TABLE {table_id};"
        await self.execute_query(truncate_table_query)
        logging.info("Truncated table %s", table_name)


def execute_with_retries(db_func: Callable[[], Any]):
    """
    Retry a function that interacts with BigQuery if it fails with a NotFound error

    Args:
        db_func (function): function that interacts with BigQuery
    """
    retries = 0
    max_retries = 5
    while retries < max_retries:
        try:
            db_func()
            break
        except Exception as e:  # TODO: more specific?
            if retries < max_retries - 1:  # don't wait on last iteration
                time.sleep(1 * retries + 1)  # backoff
            retries += 1
