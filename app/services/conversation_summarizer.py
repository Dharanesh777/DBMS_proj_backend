import json
import logging
import os
from typing import Dict, Any, Tuple
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

logger = logging.getLogger(__name__)

_openai_client = None
_cached_provider = None

def get_llm_client() -> tuple[OpenAI, str]:
    """Return a configured LLM client and the provider name.
    Supported providers: 'openai', 'groq', 'ollama'.
    Recreates the client whenever LLM_PROVIDER changes at runtime.
    """
    global _openai_client, _cached_provider
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if _openai_client is None or provider != _cached_provider:
        logger.info(f"[LLM] Building new summarizer client for provider: {provider}")
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set in environment variables.")
            _openai_client = OpenAI(api_key=api_key)
        elif provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY is not set in environment variables.")
            _openai_client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        elif provider == "ollama":
            _openai_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
        _cached_provider = provider
    return _openai_client, provider


def reset_llm_client():
    """Force the client to be recreated on next call (used after provider switch)."""
    global _openai_client, _cached_provider
    _openai_client = None
    _cached_provider = None

def analyze_conversation(transcript: str, current_time: datetime = None) -> Dict[str, Any]:
    """
    Analyzes a conversation transcript to:
    1. Generate a 3-5 line summary.
    2. Extract any discussed calendar events into specific JSON structures.
    
    Returns:
        Dict spanning the "summary" string and the "events" list.
    """
    if not transcript or not transcript.strip():
        return {"summary": "No conversation detected.", "events": []}

    if current_time is None:
        current_time = datetime.now()

    prompt = f"""
    You are an intelligent assistant analyzing a conversation transcript.
    Today's current date and time is: {current_time.strftime('%Y-%m-%d %H:%M:%S')}
    
    Please read the following conversation:
    "{transcript}"

    Output EXACTLY a JSON object with three keys: "summary", "emotion", and "events".
    
    1. "summary": Provide a 3 to 5 line summary of the conversation. 
    2. "emotion": A one or two-word string representing the overall emotional tone (e.g. Happy, Anxious, Neutral).
    3. "events": A list of important events/appointments discussed. If there are none, return an empty list [].
    
    For each event in the "events" array, it MUST follow exactly this format (use 24-hour time):
    {{
      "title": "Meeting with team",
      "date": "2026-04-20",
      "time": "10:30"
    }}
    
    Only return valid JSON. Do not return markdown blocks like ```json
    """

    client, provider = get_llm_client()

    # Choose model based on provider
    if provider == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    elif provider == "groq":
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    elif provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "llama3")
    else:
        model = "gpt-4o-mini"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2, # Low temp for structured extraction
        )
        
        content = response.choices[0].message.content
        data = json.loads(content)
        
        # Ensure expected keys exist
        data.setdefault("summary", "No summary generated.")
        data.setdefault("emotion", "Neutral")
        data.setdefault("events", [])
            
        return data

    except Exception as e:
        logger.error(f"Failed to analyze conversation: {e}")
        return {"summary": "Failed to run summarization analysis due to an error.", "events": []}



# ==========================================
# Self-Test Execution Block
# ==========================================
if __name__ == "__main__":
    test_transcript = "Hey, it was great catching up. Let's make sure we sync with the engineering team tomorrow at 2 PM. Also, don't forget the dentist appointment on April 25th at 9:15 AM."
    
    print("\n--- Testing Conversation Analyzer ---")
    print(f"Transcript: {test_transcript}\n")
    
    try:
        result_data = analyze_conversation(test_transcript)
        
        print("💡 COMBINED JSON RESULT:")
        print(json.dumps(result_data, indent=2))
        
    except ValueError as e:
        print(f"❌ Error: {e}")
