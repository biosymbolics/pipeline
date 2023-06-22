### Setup

- `brew install pre-commit`
- `pre-commit install`
- `. .pythonenv`
- `source .env` (containing AIRTABLE_API_KEY, OPENAI_API_KEY, SEC_API_KEY)
- `python3 -m pip install -r requirements.txt`

### Running

#### UI

- `streamlit run src/app/Hello.py`

#### SEC Workflow

- Edit ticker in sec_workflow.py
- Ensure airtable table exists
- `python3 src/workflows/enpv/sec_workflow.py`
- See in [Airtable](https://airtable.com/appcXwgAM75mx9sGi/tblc8ZB3S1n2NY8r9/viwKuDHk9YzcF95pd?blocks=hide)

### Testing

- Run `python3 -m pytest src`

### Static checks

- `pre-commit run --all-files`

#### Lint

- `python3 -m pylint src/`


### Code Standards

*(Notes for self as I attempt to be more disciplined)*

- Static typing, yay!
- functional, but some simple classes
- Method naming
  - fetch_xyz: pulling data from an API
    - similarly, create_xyz, update_xyz, upsert_xyz, remove_xyz
  - load_xyz: load something from local storage, or once API connection established
  - get_xyz: get a thing already fetched/loaded
  - build_xyz: build up something like an index
  - persist_xyz: save it
  - query_xyz: query a thing (esp when fetch_xyz isn't suitable)
  - extract_xyz: e.g. extract entity xyz from a document
  - format_xyz: presentation-layer formatting of xyz
  - x_to_y: conversion of X to Y
  - run_xyz: execute a workflow
  - {informative_verb}_xyz: e.g. visualize_network
  - modifiers
    - maybe_{verb}_xyz: maybe do the thing (e.g. maybe_build_xyz if it won't be built if already existing)
    - and/or: create_and_query_xyz, get_or_create_xyz
- Function overloading using @dispatch
- Runtime validation of inputs with jsonschema
- Imports
  - dot import for siblings (e.g. `from .llama_index import load_index`)
- Exports
  - export 'public' methods in __init__.py
- String formatting
  - f-strings in almost all cases (e.g. `f"{a_variable} and some text"`)
  - ... except of logging, and then `%s` (e.g. `logging.error("Error doing thing: %s", error)`)


### On Dependencies
- using python3.10 due to nmslib (dep of scispacy) along the lines of https://github.com/nmslib/nmslib/issues/476
- polars > pandas

### Misc

#### Debug Logging in REPL
```
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
```

#### Reloading module in REPL
```
from system import initialize
initialize()
import importlib
importlib.reload(core)
```

#### Other
```
ns = dict_to_named_tuple({
    "company": "PFE",
    "doc_source": "SEC",
    "doc_type": "10-K",
    "period": "2020-12-31",
})
si = core.indices.source_doc_index.SourceDocIndex()
```
