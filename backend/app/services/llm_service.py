"""
services/llm_service.py — OpenAI GPT integration for summarization
"""
import asyncio
import logging
from openai import AsyncOpenAI
from openai import APIError, APITimeoutError, RateLimitError

from ..config import get_settings

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM-powered summarization"""

    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)

    async def summarize_session(
        self,
        transcript: str,
        user_context: str | None = None,
        person_relationship: str | None = None,
    ) -> dict:
        """
        Summarize a single 30-minute session transcript and detect emotion.
        
        Returns:
            dict with 'summary' and 'emotion'
        """
        prompt = self._build_session_summary_prompt(
            transcript, user_context, person_relationship
        )
        
        response_text = await self._call_llm_with_retry(
            prompt=prompt,
            max_tokens=200,
            temperature=0.3,
        )
        return self._parse_structured_response(response_text)

    async def merge_session_summaries(
        self,
        session_summaries: list[str],
        user_context: str | None = None,
    ) -> dict:
        """
        Merge multiple session summaries into one interaction summary.
        
        Returns:
            dict with 'summary' and 'emotion'
        """
        # If only one session, it might already be a summary string or a dict-like string
        # But here we expect the list of strings gathered from summarize_session
        
        prompt = self._build_merge_summary_prompt(session_summaries, user_context)
        
        response_text = await self._call_llm_with_retry(
            prompt=prompt,
            max_tokens=400,
            temperature=0.3,
        )
        return self._parse_structured_response(response_text)

    def _build_session_summary_prompt(
        self,
        transcript: str,
        user_context: str | None,
        person_relationship: str | None,
    ) -> str:
        """Build prompt for session summarization"""
        context_info = ""
        if user_context:
            context_info += f"\n\nUser context: {user_context}"
        if person_relationship:
            context_info += f"\nRelationship: {person_relationship}"
        
        return f"""You are summarizing a 30-minute conversation for a person with short-term memory loss.{context_info}

Conversation transcript:
{transcript}

Provide a concise summary (100 words or fewer) and detect the overall emotional tone.
Format your response exactly like this:
EMOTION: [One or two words, e.g., Happy, Anxious, Neutral]
SUMMARY: [Your summary here]"""

    def _build_merge_summary_prompt(
        self,
        session_summaries: list[str],
        user_context: str | None,
    ) -> str:
        """Build prompt for merging session summaries"""
        context_info = f"\n\nUser context: {user_context}" if user_context else ""
        
        combined = "\n\n".join([
            f"Part {i+1}: {summary}"
            for i, summary in enumerate(session_summaries)
        ])
        
        return f"""You are merging summaries of a continuous conversation for a person with short-term memory loss.{context_info}

These are summaries of 30-minute parts of one visit:
{combined}

Merge into a single coherent summary (200 words or fewer) and detect the overall emotional tone.
Format your response exactly like this:
EMOTION: [One or two words, e.g., Happy, Anxious, Neutral]
SUMMARY: [Your merged summary here]"""

    def _parse_structured_response(self, text: str) -> dict:
        """Parse the EMOTION/SUMMARY format from LLM response"""
        emotion = "Neutral"
        summary = text
        
        if "EMOTION:" in text and "SUMMARY:" in text:
            try:
                parts = text.split("SUMMARY:", 1)
                emotion_part = parts[0].replace("EMOTION:", "").strip()
                summary_part = parts[1].strip()
                
                if emotion_part:
                    emotion = emotion_part
                if summary_part:
                    summary = summary_part
            except Exception:
                pass
                
        return {"summary": summary, "emotion": emotion}

    async def _call_llm_with_retry(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """
        Call OpenAI API with exponential backoff retry logic.
        
        Returns:
            Generated text or fallback message on failure
        """
        max_retries = self.settings.LLM_MAX_RETRIES
        timeout = self.settings.LLM_TIMEOUT_SECONDS
        
        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.settings.OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that summarizes conversations concisely."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    ),
                    timeout=timeout
                )
                
                return response.choices[0].message.content.strip()
            
            except (APITimeoutError, asyncio.TimeoutError) as e:
                logger.warning(f"LLM timeout on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
            
            except RateLimitError as e:
                logger.warning(f"LLM rate limit on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue
            
            except APIError as e:
                logger.error(f"LLM API error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
            
            except Exception as e:
                logger.error(f"Unexpected LLM error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
        
        # All retries failed
        logger.error("All LLM retry attempts failed")
        return "[Summary generation failed - please review transcript manually]"
