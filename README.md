### Setup

- `brew install pre-commit`
- `pre-commit install`
- `. .pythonenv`
- `source .env` (containing AIRTABLE_API_KEY, OPENAI_API_KEY, SEC_API_KEY)
- `python3 -m pip install -r requirements.txt`

### Running

#### SEC Workflow

- Edit ticker in sec_workflow.py
- Ensure airtable table exists
- `python3 src/workflows/enpv/sec_workflow.py`
- See in [Airtable](https://airtable.com/appcXwgAM75mx9sGi/tblc8ZB3S1n2NY8r9/viwKuDHk9YzcF95pd?blocks=hide)

### Testing

- Run `python3 -m pytest src`

### Static checks

- `pre-commit run --all-files`
