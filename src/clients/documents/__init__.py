from .approvals import search as approval_client
from .patents import search as patent_client
from .trials import search as trial_client

__all__ = ["approval_client", "patent_client", "trial_client"]
