"""
Utils for validation
"""

import logging
from jsonschema import validate

from common.utils.file import save_as_pickle


def validate_or_pickle(obj, schema):
    """
    Validate obj against schema.
    If failing, save pickled object for inspection
    """
    try:
        validate(instance=obj, schema=schema)
    except Exception as ex:
        pickle_file = save_as_pickle(obj)
        logging.error("Validation failure: %s (pickled + saved %s)", ex, pickle_file)
        raise ex
