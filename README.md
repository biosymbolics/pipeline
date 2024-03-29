# Biosymbolics Models and APIs

Backend for Biosymbolics - Pattern-based discovery of assets and trends in biopharma.

See also: [Biosymbolics UI](https://github.com/biosymbolics/ui)


## The Gist

A platform to support pattern-based discovery of assets across:
 - therapeutic area, treatment mechanism and modality
 - basic science, patents, clinical trials, regulatory approvals and market access

and for discovering latent predictors of success (patent signals, company performance, publication trends, mechanistic insights).

![4fb74a16-553d-4224-86a5-396b700e2165_rw_1920](https://github.com/biosymbolics/pipeline/assets/9382486/a0f40593-222f-4db2-b8a7-97d99c26a8a6)

### Data Schema
Simplified ERD
![Untitled (1)](https://github.com/biosymbolics/pipeline/assets/9382486/7443ba7a-4b02-4681-b36c-c47cb1e3bf47)

### Running locally
```
aws configure sso # url: https://d-90679e1dba.awsapps.com/start#/; CLI profile name: default
. .pythonenv
export DATABASE_URL=...
sls offline
```

### Deploy
```
$ serverless deploy
```

#### Copying data to AWS
```
# from local machine
pg_dump --no-owner biosym > biosym.psql
zip biosym.psql.zip biosym.psql
aws s3 mv s3://biosympatentsdb/biosym.psql.zip s3://biosympatentsdb/biosym.psql.zip.back-$(date +%Y-%m-%d)
aws s3 cp biosym.psql.zip s3://biosympatentsdb/biosym.psql.zip
rm biosym.psql*

# then proceeding in ec2
aws configure sso
aws s3 cp s3://biosympatentsdb/biosym.psql.zip biosym.psql.zip
unzip biosym.psql.zip
y
export PASSWORD=$(aws ssm get-parameter --name /biosymbolics/pipeline/database/patents/main_password --with-decryption --query Parameter.Value --output text)
echo "
CREATE ROLE readaccess;
GRANT USAGE ON SCHEMA public TO readaccess;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readaccess;
GRANT CONNECT ON DATABASE patents TO readaccess;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO readaccess;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO readaccess;
CREATE USER patents with password '$PASSWORD';
GRANT readaccess TO patents;
create extension vector; -- todo: move to beginning
    " >> biosym.psql
echo $PASSWORD
dropdb biosym --force  -h 172.31.55.68 -p 5432 --username postgres
createdb biosym -h 172.31.55.68 -p 5432 --username postgres
psql -d biosym -h 172.31.55.68 -p 5432 --username postgres --password -f biosym.psql
rm biosym.psql*
```

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


### Dependencies

### Updating
```
python3 -m pip freeze | grep -v @ | awk -F'==' '{print $1}' | xargs python3 -m pip install --upgrade
```

### Misc

#### Debug Logging in REPL
```
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
```

#### Profiling
```
python3 -m cProfile -o profile2.out -m data.etl.entity.biomedical_entity.biomedical_entity_load --force_update

import pstats
p = pstats.Stats('/Users/kristinlindquist/development/pipeline/profile2.out')
p.sort_stats('time').print_stats(100)

```

#### Debugging memory leak
I think there is a memory leak somewhere around spacy transformers, or in this code.
Related? https://github.com/explosion/spaCy/discussions/12093
Debugging:
```
gc.set_debug(gc.DEBUG_UNCOLLECTABLE)
import tracemalloc

tracemalloc.start()

# ... run your application ...

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("[ Top 10 ]")
for stat in top_stats[:10]:
    print(stat)
```

#### Database
```
export DATABASE_URL=postgres://biosym:ok@localhost:5432/biosym
create role biosym with password 'ok';
alter role biosym with superuser;
prisma db push

npx prisma migrate diff --from-empty --to-schema-datamodel prisma/schema.prisma --script > prisma/generated.sql
```

##### Ctgov
```
createdb aact
pg_restore -e -v -O -x -d aact --no-owner ~/Downloads/postgres_data.dmp
alter database aact set search_path = ctgov, public;
create extension vector;
create table trial_vectors(id text, vector vector(768));
```


## Training Models

### Setup on linux to train NER model
```
git clone https://github.com/biosymbolics/pipeline
apt-get install gcc g++ unzip screen nvtop
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-460.0.0-linux-x86_64.tar.gz
tar -xf google-cloud-cli-460.0.0-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh
source ~/.bashrc
gcloud auth application-default login
pip install -U prisma
cd pipeline
python3 -m pip install -r requirements.txt
python -m spacy download en_core_web_trf
prisma generate
mkdir models
mkdir data
pip uninstall torch
pip install torch torchvision torchaudio
pip install pandas
pip install nvidia-smi
pip install cupy-cuda12x
export DEFAULT_TORCH_DEVICE=cuda
cp .env.template .env
. .pythonenv
PYTHONPATH=$(pwd)/src:$(pwd)/:./binder:/opt/homebrew/lib/python3.11/site-packages/
```
- gcloud auth application-default login
- aws configure sso, using url https://d-90679e1dba.awsapps.com/start#/
- gpu activity `nvtop`
- `screen` - `ctrl-a d` to detach, `screen -r`` to reattach
- `python3 -m scripts.stage1_patents.extract_entities WO-2004078070-A3` (or whatever)
