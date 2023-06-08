from typing import Callable, Union
from llama_index import Document

DocMetadata = dict[
    str, Union[str, float, int]
]  # Pinecone docs say more types are okay?
GetDocMetadata = Callable[[Document], DocMetadata]

GetDocId = Callable[[Document], str]
