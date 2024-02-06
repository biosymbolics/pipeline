"""
Report constants
"""

from typings.documents.common import DocType

from .types import DimensionInfo

X_DIMENSIONS: dict[DocType, dict[str, DimensionInfo]] = {
    DocType.regulatory_approval: {
        "name": DimensionInfo(is_entity=True),
        "canonical_name": DimensionInfo(is_entity=True),
        "instance_rollup": DimensionInfo(is_entity=True),
        "category_rollup": DimensionInfo(is_entity=True),
    },
    DocType.patent: {
        "attributes": DimensionInfo(),
        "name": DimensionInfo(is_entity=True),
        "canonical_name": DimensionInfo(is_entity=True),
        "instance_rollup": DimensionInfo(is_entity=True),
        "category_rollup": DimensionInfo(is_entity=True),
        "similar_patents": DimensionInfo(),
    },
    DocType.trial: {
        "name": DimensionInfo(is_entity=True),
        "canonical_name": DimensionInfo(is_entity=True),
        "instance_rollup": DimensionInfo(is_entity=True),
        "category_rollup": DimensionInfo(is_entity=True),
    },
}

Y_DIMENSIONS: dict[DocType, dict[str, DimensionInfo]] = {
    DocType.regulatory_approval: {
        "id": DimensionInfo(transform=lambda y: f"regulatory_approval.{y}"),
        "approval_date": DimensionInfo(transform=lambda y: f"DATE_PART('Year', {y})"),
    },
    DocType.patent: {
        "id": DimensionInfo(transform=lambda y: f"patent.{y}"),
        "country_code": DimensionInfo(),
        "priority_date": DimensionInfo(transform=lambda y: f"DATE_PART('Year', {y})"),
    },
    DocType.trial: {
        "id": DimensionInfo(transform=lambda y: f"trial.{y}"),
        "comparison_type": DimensionInfo(),
        "design": DimensionInfo(),
        "end_date": DimensionInfo(transform=lambda y: f"DATE_PART('Year', {y})"),
        "hypothesis_type": DimensionInfo(),
        "masking": DimensionInfo(),
        "start_date": DimensionInfo(transform=lambda y: f"DATE_PART('Year', {y})"),
        "status": DimensionInfo(),
        "termination_reason": DimensionInfo(),
    },
}
