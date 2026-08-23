"""
models/user.py — ORM model for public.users
Schema columns: userid, name, age, medicalcondition, emergencycontact, email, createdat

The google_token_json column (from the removed DB-backed Google OAuth path,
see calendar_service.py / note_service.py) was dropped via Alembic migration
74bf5b968796.
"""
from datetime import datetime
from sqlalchemy import Integer, String, Text, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base


class User(Base):
    __tablename__ = "users"
    __table_args__ = {"schema": "public"}

    userid: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str | None] = mapped_column(String(100))
    age: Mapped[int | None] = mapped_column(Integer)
    medicalcondition: Mapped[str | None] = mapped_column(Text)
    emergencycontact: Mapped[str | None] = mapped_column(String(20))
    email: Mapped[str | None] = mapped_column(String(150), unique=True)
    createdat: Mapped[datetime | None] = mapped_column(
        TIMESTAMP, default=datetime.utcnow
    )

    # Relationships
    conversations = relationship("Conversation", back_populates="user")
    calendar_events = relationship("CalendarEvent", back_populates="user")
    known_persons = relationship(
        "KnownPerson",
        secondary="public.userknownperson",
        back_populates="users",
    )
    caregivers = relationship(
        "Caregiver",
        secondary="public.usercaregiver",
        back_populates="users",
    )

    def __repr__(self) -> str:
        return f"<User userid={self.userid} name={self.name!r}>"
