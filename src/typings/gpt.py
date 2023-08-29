from abc import abstractmethod
from typing import Any, Protocol


class OutputParser(Protocol):
    """
    Output parser class.
    Copied from langchain to avoid import
    """

    @abstractmethod
    def parse(self, output: str) -> Any:
        """Parse, validate, and correct errors programmatically."""

    @abstractmethod
    def format(self, output: str) -> str:
        """Format a query with structured output formatting instructions."""
