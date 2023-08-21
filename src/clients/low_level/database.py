from abc import abstractmethod
import logging
import time
from typing import Any, Callable, Mapping, TypeVar, TypedDict
import logging

from utils.list import batch

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Mapping)
ExecuteResult = TypedDict("ExecuteResult", {"columns": list[str], "data": list[tuple]})


class DatabaseClient:
    def __init__(self):
        self.client = None  # Override

    @staticmethod
    @abstractmethod
    def get_table_id(table_name: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def execute_query(self, query: str, values: list = []) -> ExecuteResult:
        raise NotImplementedError

    @abstractmethod
    def is_table_exists(self, table_name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _insert(self, table_name: str, records: list[T]):
        raise NotImplementedError

    @abstractmethod
    def _create(
        self,
        table_name: str,
        columns: list[str] | dict[str, str],
    ):
        raise NotImplementedError

    def create_from_select(self, query: str, new_table_name: str):
        """
        Create a new table from a query

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
        self.execute_query(create_table_query)

    def select(self, query: str, values: list = []) -> list[dict]:
        """
        Execute a query and return the results as a list of dicts
        (must include provide fully qualified table name in query)

        Args:
            query (str): SQL query
        """
        logger.debug("Running query: %s", query)
        results = self.execute_query(query, values)
        records = [dict(row) for row in results["data"]]

        logging.info("Records returned: %s", len(records))
        return records

    def select_insert_into_table(self, select_query: str, table_name: str):
        """
        Insert rows into a table from a select query

        Args:
            select_query (str): select query
            table_name (str): name of the table
        """
        table_id = self.get_table_id(table_name)
        query = f"INSERT INTO {table_id} {select_query}"
        logger.info("Inserting via query (%s) into table %s", query, table_id)
        self.execute_query(query)

    def insert_into_table(
        self, records: list[T], table_name: str, batch_size: int = 1000
    ):
        """
        Insert rows into a table from a list of records

        Args:
            records (list[dict]): list of records to insert
            table_name (str): name of the table
            batch_size (int, optional): number of records to insert per batch. Defaults to 1000.
        """
        batched = batch(records, batch_size)

        for i, b in enumerate(batched):
            logging.info("Inserting batch %s into table %s", i, table_name)
            try:
                self._insert(table_name, records=b)
            except Exception as e:
                logging.error("Error inserting rows: %s", e)
                raise e

            logging.info("Successfully inserted %s rows", len(b))

    def create_table(
        self,
        table_name: str,
        columns: list[str] | dict[str, str],
        exists_ok: bool = True,
        truncate_if_exists: bool = False,
    ):
        """
        Create a table

        Args:
            table_name (str): name of the table (with or without dataset prefix)
            columns: list of column names or dict (column name -> type)
            exists_ok (bool): if True, do not raise an error if the table already exists
            truncate_if_exists (bool): if True, truncate the table if it already exists
        """
        logging.info("Creating table: %s", table_name)

        if truncate_if_exists and not exists_ok:
            raise Exception("Cannot truncate if exists if exists_ok is False")

        if self.is_table_exists(table_name):
            if truncate_if_exists:
                self.truncate_table(table_name)
            elif not exists_ok:
                raise Exception(f"Table {table_name} already exists")
            else:
                logging.info("Table %s already exists", table_name)
        else:
            self._create(table_name, columns)

    def delete_table(self, table_name: str):
        """
        Delete a table (if exists)

        Args:
            table_name (str): name of the table
        """
        table_id = self.get_table_id(table_name)
        logger.info("Deleting table %s", table_name)
        delete_table_query = f"DROP TABLE IF EXISTS {table_id};"
        self.execute_query(delete_table_query)

    def truncate_table(self, table_name: str):
        """
        Truncate a table (if exists)

        Args:
            table_name (str): name of the table
        """
        exists = self.is_table_exists(table_name)
        if not exists:
            logger.warning(
                "Table %s does not exist and thus not truncating", table_name
            )
            return

        table_id = self.get_table_id(table_name)
        logger.info("Truncating table %s", table_id)
        truncate_table_query = f"TRUNCATE TABLE {table_id};"
        self.execute_query(truncate_table_query)


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
