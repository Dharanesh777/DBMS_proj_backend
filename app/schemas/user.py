"""
schemas/user.py — Pydantic schemas for User endpoints
"""
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class UserBase(BaseModel):
    """Base user schema"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    age: Optional[int] = Field(None, ge=0, le=150)
    medicalcondition: Optional[str] = None
    emergencycontact: Optional[str] = Field(None, max_length=20)
    # max_length matches users.email VARCHAR(150) — without it, a >150-char
    # valid email passes Pydantic and then throws a raw Postgres DataError
    # (surfaced as an ugly 500) instead of a clean 400.
    email: Optional[EmailStr] = Field(None, max_length=150)


class UserCreate(UserBase):
    """Schema for creating a new user"""
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr = Field(..., max_length=150)


class UserUpdate(UserBase):
    """Schema for updating user (all fields optional)"""
    pass


class UserResponse(UserBase):
    """Schema for user response.

    email is widened back to a plain str (not EmailStr) here — this serializes
    EXISTING rows, and strict format validation on read means one legacy row
    with an atypical email (e.g. a .local domain, which email-validator treats
    as a reserved TLD) would 500 the entire list for every caller. Format is
    still enforced on write via UserCreate.email: EmailStr.
    """
    userid: int
    email: Optional[str] = None
    createdat: Optional[datetime] = None

    class Config:
        from_attributes = True


class UserListResponse(BaseModel):
    """Schema for listing users"""
    users: list[UserResponse]
    total: int
