import torch

# from transformers import MarkupLMProcessor
from transformers import (
    pipeline,
    AutoTokenizer,
    MarkupLMFeatureExtractor,
    MarkupLMTokenizerFast,
    MarkupLMProcessor,
    MarkupLMForTokenClassification,
    AutoProcessor,
    AutoModelForTokenClassification,
)


def query_html(html_string: str, question: str) -> str:
    """
    Queries an HTML string with natural language

    Args:
        html_string (str): HTML string to query
        query (str): Query to ask
    """
    processor = AutoProcessor.from_pretrained(
        "microsoft/markuplm-base-finetuned-websrc"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/markuplm-base-finetuned-websrc"
    )
    # processor.parse_html = False
    model = AutoModelForTokenClassification.from_pretrained(
        "microsoft/markuplm-base-finetuned-websrc"
    )

    encoding = processor(return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encoding)

    nerpipeline = pipeline("ner", model=model, tokenizer=tokenizer)
    things = nerpipeline(html_string)
    print(things)

    # with torch.no_grad():
    #     outputs = model(**encoding)
    #     print(outputs)

    #     loss = outputs.loss
    #     logits = outputs.logits

    return ""
