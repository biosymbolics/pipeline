from typing import TypedDict
import logging

from clients.openai.gpt_client import GptApiClient


class ClinDevParams(TypedDict):
    indication: str


class ClinDevEvent(TypedDict):
    queryStringParameters: ClinDevParams


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def predict_timelines(event: ClinDevEvent, context):
    """
    Query GPT about clindev timelines

    Invocation:
    - Local: `serverless invoke local --function predict-clindev --data='{"queryStringParameters": { "indication": "asthma" }}'`
    - Remote: `serverless invoke --function predict-clindev --data='{"queryStringParameters": { "indication": "asthma" }}'`
    - API: `curl https://api.biosymbolics.ai/clindev/predict/timelines?indication=asthma`
    """
    gpt_client = GptApiClient(model="gpt-3.5-turbo")

    params = event.get("queryStringParameters", {})
    indication = params.get("indication")

    if not indication or len(indication) < 5:
        logger.error(
            "Missing or malformed query params: %s",
            params,
        )
        return {
            "statusCode": 400,
            "body": "Missing parameter(s)",
        }

    logger.info(
        "Fetching info for indication: %s",
        indication,
    )

    try:
        answer = gpt_client.clindev_timelines(indication)
    except Exception as e:
        logger.error("Error fetching info for indication: %s", e)
        return {"statusCode": 500, "body": "Error fetching info for indication"}

    return {"statusCode": 200, "body": answer}
