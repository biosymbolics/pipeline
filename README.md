### Setup

- `brew install pre-commit`
- `pre-commit install`
- `. .pythonenv`
- `source .env` (containing OPENAI_API_KEY, SEC_API_KEY, etc)
- `python3 -m pip install -r requirements.txt`
- `python3 -m spacy download en_core_web_md`
- Create/copy binder.pt for non-GPT NER
- AWS authenticate if needed: `aws configure sso`
- psql
  - psql -h 172.31.14.226 -p 5432 --username postgres
- sls offlne
  - https://github.com/dherault/serverless-offline/issues/1696
- Docker - new ECR for lambda
```
aws ecr create-repository --repository-name biosym_lambda-repo

docker build --platform linux/amd64 -t biosym_lambda-repo:chat_torch_latest .
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 469840476741.dkr.ecr.us-east-1.amazonaws.com
docker tag biosym_lambda-repo:chat_torch_latest 469840476741.dkr.ecr.us-east-1.amazonaws.com/biosym_lambda-repo:chat_torch_latest
docker push 469840476741.dkr.ecr.us-east-1.amazonaws.com/biosym_lambda-repo:chat_torch_latest
```

### Running locally
- `sls offline`

### Deploy

In order to deploy the example, you need to run the following command:

```
$ serverless deploy
```

#### Invocation

After successful deployment, you can invoke the deployed function by using the following command:

**Remote**:
```bash
serverless invoke --function search-patents
```

**Local**:
```bash
serverless invoke local --function search-patents
```

### Running

#### Patents

##### NER

- `. .pythonenv`
- `source ~/.bashrc`
- `gcloud auth application-default login`
- `python3 src/scripts/patents/ner.py`

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

#### Dependencies
```
python3 -m pip freeze | grep -v @ | awk -F'==' '{print $1}' | xargs python3 -m pip install --upgrade
```

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
importlib.reload(common.ner.ner)
```

#### Redis
- `redis-cli -h redis-12973.c1.us-west-2-2.ec2.cloud.redislabs.com -p 12973`
- `FLUSHALL`
- `keys *`
- `HGETALL "term:PD-1 inhibitors"`

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

#### Test
```
import system
system.initialize()
from core.ner import NerTagger
tagger = NerTagger()
text = "Asthma may be associated with Parkinson's disease and treated with SHAI inhibitors)."
tagger.extract([text])
```
