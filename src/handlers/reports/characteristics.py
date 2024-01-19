"""
Handler for patent graph reports
"""
import json
import logging
from clients.documents.constants import DOC_CLIENT_LOOKUP

from clients.documents.reports.graph import aggregate_document_relationships
from handlers.utils import handle_async
from typings.client import CommonSearchParams
from typings.documents.common import DocType
from utils.encoding.json_encoder import DataclassJSONEncoder

from .constants import DEFAULT_REPORT_PARAMS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# TODO: non-patent-specific
class DocumentCharacteristicParams(CommonSearchParams):
    """
    Parameters for document characteristics
    """

    doc_type: DocType = DocType.patent
    head_field: str = "priority_date"


async def _document_characteristics(raw_event: dict, context):
    """
    Return a graph of document characteristics

    Invocation:
    - Local: `serverless invoke local --function document-characteristics --param='ENV=local' --data='{"queryStringParameters": { "terms":"gpr84 antagonist" }}'`
    - Remote: `serverless invoke --function document-characteristics --data='{"queryStringParameters": { "terms":"gpr84 antagonist" }}'`
    - API: `curl https://api.biosymbolics.ai/reports/graph?terms=asthma`
    """
    p = DocumentCharacteristicParams(
        **{**raw_event["queryStringParameters"], "include": {}, **DEFAULT_REPORT_PARAMS}
    )
    if len(p.terms) < 1 or not all([len(t) > 1 for t in p.terms]):
        logger.error("Missing or malformed params: %s", p)
        return {"statusCode": 400, "body": "Missing params(s)"}

    logger.info("Fetching reports for params: %s", p)

    try:
        documents = await DOC_CLIENT_LOOKUP[p.doc_type].search(p)

        if len(documents) == 0:
            logging.info("No documents found for terms: %s", p.terms)
            return {"statusCode": 200, "body": json.dumps([])}

        ids = [d.id for d in documents]
        report = await aggregate_document_relationships(ids, p.head_field)
    except Exception as e:
        message = f"Error generating patent reports: {e}"
        logger.error(message)
        return {"statusCode": 500, "body": message}

    return {
        "statusCode": 200,
        "body": json.dumps(report, cls=DataclassJSONEncoder),
    }


document_characteristics = handle_async(_document_characteristics)
