"""
services/caregiver_service.py — Business logic for Caregiver management
"""
import logging
from sqlalchemy.orm import Session
from sqlalchemy import select, insert, delete, func
from typing import Optional

from app.models.caregiver import Caregiver
from app.models.user import User
from app.models.junction_tables import usercaregiver

logger = logging.getLogger(__name__)


class CaregiverService:
    """Service for managing caregivers"""

    def __init__(self, db: Session):
        self.db = db

    def create_caregiver(
        self,
        name: str,
        relationshiptouser: str,
        accesslevel: str = "read",
    ) -> Caregiver:
        """
        Create a new caregiver.
        
        Args:
            name: Caregiver's name
            relationshiptouser: Relationship to user (e.g., "spouse", "child", "nurse")
            accesslevel: Access level (e.g., "read", "write", "admin")
        
        Returns:
            Created Caregiver object
        """
        caregiver = Caregiver(
            name=name,
            relationshiptouser=relationshiptouser,
            accesslevel=accesslevel,
        )
        
        self.db.add(caregiver)
        self.db.commit()
        self.db.refresh(caregiver)
        
        logger.info(f"Created caregiver {caregiver.caregiverid}: {name}")
        return caregiver

    def get_caregiver(self, caregiver_id: int) -> Optional[Caregiver]:
        """Get caregiver by ID, with no ownership scoping. Internal use only —
        route handlers must use get_caregiver_for_user instead."""
        return self.db.execute(
            select(Caregiver).where(Caregiver.caregiverid == caregiver_id)
        ).scalar_one_or_none()

    def get_caregiver_for_user(self, caregiver_id: int, user_id: int) -> Optional[Caregiver]:
        """Get a caregiver by ID, scoped to caregivers actually assigned to user_id."""
        return self.db.execute(
            select(Caregiver)
            .join(usercaregiver, Caregiver.caregiverid == usercaregiver.c.caregiverid)
            .where(
                Caregiver.caregiverid == caregiver_id,
                usercaregiver.c.userid == user_id,
            )
        ).scalar_one_or_none()

    def list_caregivers_for_user(self, user_id: int, skip: int = 0, limit: int = 100) -> list[Caregiver]:
        """List caregivers assigned to user_id, with pagination."""
        return list(
            self.db.execute(
                select(Caregiver)
                .join(usercaregiver, Caregiver.caregiverid == usercaregiver.c.caregiverid)
                .where(usercaregiver.c.userid == user_id)
                .order_by(Caregiver.caregiverid)
                .offset(skip)
                .limit(limit)
            ).scalars()
        )

    def count_caregivers_for_user(self, user_id: int) -> int:
        """Count caregivers assigned to user_id."""
        return self.db.execute(
            select(func.count(Caregiver.caregiverid))
            .join(usercaregiver, Caregiver.caregiverid == usercaregiver.c.caregiverid)
            .where(usercaregiver.c.userid == user_id)
        ).scalar_one()

    def update_caregiver(
        self,
        caregiver_id: int,
        user_id: int,
        name: Optional[str] = None,
        relationshiptouser: Optional[str] = None,
        accesslevel: Optional[str] = None,
    ) -> Caregiver:
        """
        Update caregiver information. Only permitted if caregiver_id is assigned to user_id.

        Args:
            caregiver_id: Caregiver ID to update
            user_id: Caller's user ID — caregiver must be assigned to this user
            name: New name (optional)
            relationshiptouser: New relationship (optional)
            accesslevel: New access level (optional)

        Returns:
            Updated Caregiver object

        Raises:
            ValueError: If caregiver not found or not assigned to user_id
        """
        caregiver = self.get_caregiver_for_user(caregiver_id, user_id)
        if not caregiver:
            raise ValueError(f"Caregiver {caregiver_id} not found for user {user_id}")

        if name is not None:
            caregiver.name = name
        if relationshiptouser is not None:
            caregiver.relationshiptouser = relationshiptouser
        if accesslevel is not None:
            caregiver.accesslevel = accesslevel

        self.db.commit()
        self.db.refresh(caregiver)

        logger.info(f"Updated caregiver {caregiver_id}")
        return caregiver

    def delete_caregiver(self, caregiver_id: int, user_id: int) -> bool:
        """
        Delete a caregiver. Only permitted if caregiver_id is assigned to user_id.

        Args:
            caregiver_id: Caregiver ID to delete
            user_id: Caller's user ID — caregiver must be assigned to this user

        Returns:
            True if deleted, False if not found for this user
        """
        caregiver = self.get_caregiver_for_user(caregiver_id, user_id)
        if not caregiver:
            return False

        self.db.delete(caregiver)
        self.db.commit()

        logger.info(f"Deleted caregiver {caregiver_id}")
        return True

    def assign_caregiver_to_user(self, user_id: int, caregiver_id: int) -> bool:
        """
        Assign a caregiver to a user.
        
        Args:
            user_id: User ID
            caregiver_id: Caregiver ID
        
        Returns:
            True if assigned successfully
        
        Raises:
            ValueError: If user or caregiver not found, or already assigned
        """
        # Verify user exists
        user = self.db.execute(
            select(User).where(User.userid == user_id)
        ).scalar_one_or_none()
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        # Verify caregiver exists
        caregiver = self.get_caregiver(caregiver_id)
        if not caregiver:
            raise ValueError(f"Caregiver {caregiver_id} not found")
        
        # Check if already assigned
        existing = self.db.execute(
            select(usercaregiver).where(
                usercaregiver.c.userid == user_id,
                usercaregiver.c.caregiverid == caregiver_id,
            )
        ).first()
        
        if existing:
            raise ValueError(f"Caregiver {caregiver_id} already assigned to user {user_id}")
        
        # Insert assignment
        self.db.execute(
            insert(usercaregiver).values(
                userid=user_id,
                caregiverid=caregiver_id,
            )
        )
        self.db.commit()
        
        logger.info(f"Assigned caregiver {caregiver_id} to user {user_id}")
        return True

    def unassign_caregiver_from_user(self, user_id: int, caregiver_id: int) -> bool:
        """
        Unassign a caregiver from a user.
        
        Args:
            user_id: User ID
            caregiver_id: Caregiver ID
        
        Returns:
            True if unassigned successfully, False if not found
        """
        result = self.db.execute(
            delete(usercaregiver).where(
                usercaregiver.c.userid == user_id,
                usercaregiver.c.caregiverid == caregiver_id,
            )
        )
        self.db.commit()
        
        if result.rowcount > 0:
            logger.info(f"Unassigned caregiver {caregiver_id} from user {user_id}")
            return True
        
        return False

    def get_caregivers_for_user(self, user_id: int) -> list[Caregiver]:
        """Get all caregivers assigned to a user"""
        user = self.db.execute(
            select(User).where(User.userid == user_id)
        ).scalar_one_or_none()
        
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        return user.caregivers
