"""
models/junction_tables.py — ORM Table objects for M:M junction tables.
usercaregiver and userknownperson have no extra columns so they are best
represented as plain Table objects (used in relationship secondarys).
"""
from sqlalchemy import Table, Column, Integer, ForeignKey
from app.db.base import Base

usercaregiver = Table(
    "usercaregiver",
    Base.metadata,
    Column("userid", Integer, ForeignKey("public.users.userid", ondelete="CASCADE"), primary_key=True),
    Column("caregiverid", Integer, ForeignKey("public.caregiver.caregiverid", ondelete="CASCADE"), primary_key=True),
    schema="public",
)

userknownperson = Table(
    "userknownperson",
    Base.metadata,
    Column("userid", Integer, ForeignKey("public.users.userid", ondelete="CASCADE"), primary_key=True),
    Column("personid", Integer, ForeignKey("public.knownperson.personid", ondelete="CASCADE"), primary_key=True),
    schema="public",
)
