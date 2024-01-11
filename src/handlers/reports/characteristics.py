"""
Handler for patent graph reports
"""
import json
import logging

from clients.documents import patents as patent_client
from clients.documents.reports.graph import aggregate_document_relationships
from clients.documents.reports.graph.types import CharacteristicHeadField
from handlers.utils import handle_async
from typings.client import PatentSearchParams
from utils.encoding.json_encoder import DataclassJSONEncoder

from .constants import DEFAULT_REPORT_PARAMS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DocumentCharacteristicParams(PatentSearchParams):
    """
    Parameters for document characteristics
    """

    head_field: CharacteristicHeadField = "priority_year"


async def _document_characteristics(raw_event: dict, context):
    """
    Return a graph of document characteristics

    Invocation:
    - Local: `serverless invoke local --function document-characteristics --param='ENV=local' --data='{"queryStringParameters": { "terms":"gpr84 antagonist" }}'`
    - Remote: `serverless invoke --function document-characteristics --data='{"queryStringParameters": { "terms":"gpr84 antagonist" }}'`
    - API: `curl https://api.biosymbolics.ai/reports/graph?terms=asthma`
    """
    p = DocumentCharacteristicParams(
        **{**raw_event["queryStringParameters"], **DEFAULT_REPORT_PARAMS}
    )
    if len(p.terms) < 1 or not all([len(t) > 1 for t in p.terms]):
        logger.error("Missing or malformed params: %s", p)
        return {"statusCode": 400, "body": "Missing params(s)"}

    logger.info("Fetching reports for params: %s", p)

    try:
        patents = await patent_client.search(p)
        if len(patents) == 0:
            logging.info("No patents found for terms: %s", p.terms)
            return {"statusCode": 200, "body": json.dumps({})}

        report = await aggregate_document_relationships(patents, p.head_field)
    except Exception as e:
        message = f"Error generating patent reports: {e}"
        logger.error(message)
        return {"statusCode": 500, "body": message}

    return {
        "statusCode": 200,
        "body": json.dumps(report, cls=DataclassJSONEncoder),
    }


document_characteristics = handle_async(_document_characteristics)
