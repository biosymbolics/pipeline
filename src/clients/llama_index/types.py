from typing import Callable, Union
from llama_index import Document

DocMetadata = dict[str, Union[str, float, int]]
GetDocMetadata = Callable[[Document], DocMetadata]
