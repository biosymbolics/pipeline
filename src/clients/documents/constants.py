from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeVar
from typings import DocType
from typings.client import DocumentSearchCriteria, DocumentSearchParams
from typings.core import Dataclass
from typings.documents.approvals import ScoredRegulatoryApproval
from typings.documents.patents import ScoredPatent
from typings.documents.trials import ScoredTrial

from . import patents as patent_client
from . import approvals as regulatory_approval_client
from . import trials as trial_client


@dataclass(frozen=True)
class DocClient(Dataclass):
    search: Callable[
        [DocumentSearchParams],
        Awaitable[
            list[ScoredPatent] | list[ScoredRegulatoryApproval] | list[ScoredTrial]
        ],
    ]
    find_many: Callable[[Any], Awaitable]


DOC_CLIENT_LOOKUP: dict[DocType, DocClient] = {
    DocType.patent: DocClient(
        search=patent_client.search,
        find_many=patent_client.find_many,
    ),
    DocType.regulatory_approval: DocClient(
        search=regulatory_approval_client.search,
        find_many=regulatory_approval_client.find_many,
    ),
    DocType.trial: DocClient(
        search=trial_client.search,
        find_many=trial_client.find_many,
    ),
}
