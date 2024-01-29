from pydantic import BaseModel
from prisma.enums import DropoutReason


class DropoutReasonRaw(BaseModel):
    reason: str
    count: int


class DropoutReasonCount(BaseModel):
    reason: DropoutReason
    count: int
