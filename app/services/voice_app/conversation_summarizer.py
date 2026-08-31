import os
import json
import urllib.request
import urllib.error
from dotenv import load_dotenv

load_dotenv()

def _get_heuristic_fallback(text: str) -> dict:
    """Return a baseline structured dict if LLM providers are unavailable."""
    clean_text = text.strip()
    summary = clean_text[:200] + ("..." if len(clean_text) > 200 else "")
    return {
        "summary": summary if summary else "No transcription content available.",
        "emotion": "Neutral",
        "events": []
    }

def analyze_conversation(text: str) -> dict:
    """
    Analyzes conversation text to generate a summary, detect emotion, and extract calendar events.
    Supports LLM providers: 'groq', 'openai', 'ollama', or falls back gracefully.
    """
    if not text or not text.strip():
        return _get_heuristic_fallback(text)

    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    
    prompt = (
        "You are an AI assistant analyzing a transcribed conversation.\n"
        "Extract the following details and return ONLY a valid JSON object without markdown formatting or code fences:\n"
        "{\n"
        '  "summary": "Brief summary of the conversation",\n'
        '  "emotion": "Overall tone or emotion (e.g. Neutral, Happy, Sad, Anxious, Enthusiastic)",\n'
        '  "events": [\n'
        '    {"title": "Event title", "date": "YYYY-MM-DD", "time": "HH:MM"}\n'
        '  ]\n'
        "}\n\n"
        f"Conversation:\n{text}"
    )

    try:
        raw_response = None

        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            if api_key:
                req = urllib.request.Request(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    data=json.dumps({
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "response_format": {"type": "json_object"}
                    }).encode("utf-8")
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    raw_response = data["choices"][0]["message"]["content"]

        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            if api_key:
                req = urllib.request.Request(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    data=json.dumps({
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "response_format": {"type": "json_object"}
                    }).encode("utf-8")
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    raw_response = data["choices"][0]["message"]["content"]

        elif provider == "ollama":
            model = os.getenv("OLLAMA_MODEL", "llama3")
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps({
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                }).encode("utf-8")
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                raw_response = data.get("response")

        if raw_response:
            cleaned = raw_response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            parsed = json.loads(cleaned.strip())
            return {
                "summary": parsed.get("summary", _get_heuristic_fallback(text)["summary"]),
                "emotion": parsed.get("emotion", "Neutral"),
                "events": parsed.get("events", [])
            }

    except Exception as e:
        print(f"[Summarizer Warning] LLM call failed or unconfigured: {e}. Falling back to baseline summary.", flush=True)

    return _get_heuristic_fallback(text)
