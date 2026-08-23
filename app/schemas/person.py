"""
schemas/person.py — Pydantic schemas for person records
"""
from pydantic import BaseModel


class PersonResponse(BaseModel):
    """Schema for person response (for listing/details)"""
    personid: int
    name: str | None = None
    relationshiptype: str | None = None
    prioritylevel: int | None = None
    notes: str | None = None

    class Config:
        from_attributes = True
