from typings import DocType

from . import patents as patent_client
from . import approvals as regulatory_approval_client
from . import trials as trial_client


DOC_CLIENT_LOOKUP = {
    DocType.patent: patent_client,
    DocType.regulatory_approval: regulatory_approval_client,
    DocType.trial: trial_client,
}
