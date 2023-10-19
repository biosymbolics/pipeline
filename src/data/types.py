from typing import TypedDict


ModelMetrics = TypedDict(
    "ModelMetrics", {"precision": float, "recall": float, "f1": float}
)
