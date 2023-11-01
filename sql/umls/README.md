# Loading UMLS

### Download
- 'UMLS Metathesaurus Level 0 Subset'
- https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html


### Load into Postgres
``` bash
unzip ~/Desktop/umls-2023AA-metathesaurus-level-0.zip
cd ~/Desktop/2023AA
dropdb umls --force
createdb umls
psql umls < ~/development/pipeline/sql/umls/pgsql_all_tables.sql
psql umls < ~/development/pipeline/sql/umls/pgsql_index.sql
```

### Schema
- https://www.ncbi.nlm.nih.gov/books/NBK9685/
