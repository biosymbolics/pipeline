"""
Patent types
"""
from typing import TypedDict

TermResult = TypedDict("TermResult", {"term": str, "count": int})
