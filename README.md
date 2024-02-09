### Setup

- `CFLAGS="-mavx -DWARN(a)=(a)" pip install 'nmslib @ git+https://github.com/nmslib/nmslib.git#egg=nmslib'`
- https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
  - https://d-90679e1dba.awsapps.com/start#/
  - CLI profile name: default
- AWS authenticate if needed: `aws configure sso`
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

### Running a model on Vast.ai (or the like)
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

### Running on server for linking
```
pg_dump --no-owner aact \
  -t studies \
  -t outcomes \
  -t designs \
  -t design_groups \
  -t browse_conditions \
  -t conditions \
  -t interventions \
  -t browse_interventions \
  -t drop_withdrawals \
  -t outcome_analyses > ct.sql
zip ct.sql.zip ct.sql
pg_dump --no-owner drugcentral \
    -t pharma_class \
    -t approval \
    -t active_ingredient \
    -t product \
    -t prd2label \
    -t label \
    -t structures \
    -t omop_relationship > drugcentral.sql
zip drugcentral.sql.zip drugcentral.sql
pg_dump --no-owner patents -t biosym_annotations -t gpr_annotations > patents.sql
zip patents.sql.zip patents.sql
aws s3 cp patents.sql.zip s3://biosym-patents-dev
### other machine
sudo apt install postgresql postgresql-contrib screen unzip gcc g++ make postgresql-server-dev-14
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
source ~/.bashrc
aws configure sso # using url https://d-90679e1dba.awsapps.com/start#/
aws s3 cp s3://biosym-patents-dev/patents.sql.zip .
git clone https://github.com/biosymbolics/pipeline
cd pipeline
python3 -m pip install -r requirements.txt
python -m spacy download en_core_web_trf
cd /tmp
git clone --branch v0.6.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
make install # may need sudo
pg_lsclusters
pg_ctlcluster 14 main start
su - postgres
psql
create role biosym with password 'ok';
alter role biosym with superuser;
ALTER ROLE "biosym" WITH LOGIN;
create database biosym;
create database patents;
create database drugcentral;
create database aact;
create schema ctgov;
\c patents;
CREATE EXTENSION vector;
exit
adduser biosym
cd /pipeline
export DATABASE_URL=postgres://biosym:ok@localhost:5432/biosym?max_connections=50
pip install -U prisma
prisma db push
unzip patents.sql.zip
unzip drugcentral.sql.zip
unzip ct.sql.zip
sudo -u biosym psql aact < ct.sql
sudo -u biosym psql drugcentral < drugcentral.sql
sudo -u biosym psql patents < patents.sql
. .pythonenv
```

### Running locally
- `sls offline`

### Deploy

In order to deploy the example, you need to run the following command:

```
$ serverless deploy
```

#### Copying data
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

#### Redis
- `redis-cli -h redis-12973.c1.us-west-2-2.ec2.cloud.redislabs.com -p 12973`
- `FLUSHALL`
- `keys *`
- `HGETALL "term:PD-1 inhibitors"`

#### Database
```
export DATABASE_URL=postgres://biosym:ok@localhost:5432/biosym
create role biosym with password 'ok';
alter role biosym with superuser;
prisma db push
```

#### Random
```
from core.ner.spacy import Spacy
nlp = Spacy.get_instance(
            model="en_core_web_trf",
            disable=["ner", "parser", "tagger"],
            additional_pipelines={
                "transformer": {
                    "config": {
                        "model": {
                            "@architectures": "spacy-transformers.TransformerModel.v3",
                            "name": "bert-base-uncased",
                            # "tokenizer_config": {"use_fast": True},
                            # "transformer_config": {},
                            # "mixed_precision": True,
                            # "grad_scaler_config": {"init_scale": 32768},
                            "get_spans": {
                                "@span_getters": "spacy-transformers.strided_spans.v1",
                                "window": 128,
                                "stride": 16,
                            },
                        },
                    },
                },
                "tok2vec": {},
            },
        )
```
