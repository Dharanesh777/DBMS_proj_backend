"""
services/emotion_service.py — Business logic for EmotionRecord management
"""
import logging
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from typing import Optional

from app.models.emotion_record import EmotionRecord
from app.models.conversation import Conversation

logger = logging.getLogger(__name__)


class EmotionService:
    """Service for managing emotion records"""

    def __init__(self, db: Session):
        self.db = db

    def create_emotion_record(
        self,
        interaction_id: int,
        emotiontype: str,
        confidencelevel: float,
    ) -> EmotionRecord:
        """
        Create a new emotion record for an interaction.
        
        Args:
            interaction_id: Interaction ID
            emotiontype: Type of emotion (e.g., "happy", "sad", "angry")
            confidencelevel: Confidence level (0.0 to 1.0)
        
        Returns:
            Created EmotionRecord object
        
        Raises:
            ValueError: If interaction not found
        """
        # Verify interaction exists
        interaction = self.db.execute(
            select(Conversation).where(Conversation.interactionid == interaction_id)
        ).scalar_one_or_none()
        
        if not interaction:
            raise ValueError(f"Interaction {interaction_id} not found")
        
        emotion = EmotionRecord(
            interactionid=interaction_id,
            emotiontype=emotiontype,
            confidencelevel=confidencelevel,
        )
        
        self.db.add(emotion)
        self.db.commit()
        self.db.refresh(emotion)
        
        logger.info(f"Created emotion record {emotion.emotionid} for interaction {interaction_id}: {emotiontype}")
        return emotion

    def get_emotion_record(self, emotion_id: int, user_id: int) -> Optional[EmotionRecord]:
        """Get emotion record by ID, scoped to interactions owned by user_id."""
        return self.db.execute(
            select(EmotionRecord)
            .join(Conversation, EmotionRecord.interactionid == Conversation.interactionid)
            .where(
                EmotionRecord.emotionid == emotion_id,
                Conversation.userid == user_id,
            )
        ).scalar_one_or_none()

    def get_emotions_for_interaction(self, interaction_id: int, user_id: int) -> list[EmotionRecord]:
        """Get all emotion records for an interaction, scoped to user_id."""
        return list(
            self.db.execute(
                select(EmotionRecord)
                .join(Conversation, EmotionRecord.interactionid == Conversation.interactionid)
                .where(
                    EmotionRecord.interactionid == interaction_id,
                    Conversation.userid == user_id,
                )
            ).scalars()
        )

    def list_emotion_records(self, user_id: int, skip: int = 0, limit: int = 100) -> list[EmotionRecord]:
        """List emotion records for interactions owned by user_id, with pagination."""
        return list(
            self.db.execute(
                select(EmotionRecord)
                .join(Conversation, EmotionRecord.interactionid == Conversation.interactionid)
                .where(Conversation.userid == user_id)
                .order_by(EmotionRecord.emotionid)
                .offset(skip)
                .limit(limit)
            ).scalars()
        )

    def count_emotion_records(self, user_id: int) -> int:
        """Count emotion records for interactions owned by user_id."""
        return self.db.execute(
            select(func.count(EmotionRecord.emotionid))
            .join(Conversation, EmotionRecord.interactionid == Conversation.interactionid)
            .where(Conversation.userid == user_id)
        ).scalar_one()

    def delete_emotion_record(self, emotion_id: int, user_id: int) -> bool:
        """
        Delete an emotion record. Only permitted if it belongs to an interaction owned by user_id.

        Args:
            emotion_id: Emotion record ID to delete
            user_id: Caller's user ID

        Returns:
            True if deleted, False if not found for this user
        """
        emotion = self.get_emotion_record(emotion_id, user_id)
        if not emotion:
            return False

        self.db.delete(emotion)
        self.db.commit()

        logger.info(f"Deleted emotion record {emotion_id}")
        return True
