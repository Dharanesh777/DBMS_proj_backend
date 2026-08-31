import os
import time
from dotenv import load_dotenv

load_dotenv()

def get_db_connection(retries=3, delay=1.0):
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError:
        raise ImportError("psycopg2 package is not installed. Install psycopg2-binary to enable DB functionality.")

    for i in range(retries):
        try:
            conn = psycopg2.connect(
                host=os.getenv("DB_HOST", "localhost"),
                port=os.getenv("DB_PORT", "5432"),
                dbname=os.getenv("DB_NAME", "postgres"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", ""),
                sslmode=os.getenv("DB_SSLMODE", "prefer"),
            )
            return conn
        except Exception as e:
            if i < retries - 1:
                print(f"[WARNING] DB Connection failed: {e}. Retrying in {delay}s... (attempt {i+1}/{retries})", flush=True)
                time.sleep(delay)
                delay *= 2
            else:
                raise e

def update_conversation_results(interactionid, transcribed_text, summarized_text, emotion_detected='Neutral'):
    try:
        conn = get_db_connection(retries=1, delay=0.5)
        cur = conn.cursor()
        query = """
            UPDATE public.conversation 
            SET conversation = %s, summarytext = %s, emotiondetected = %s
            WHERE interactionid = %s;
        """
        cur.execute(query, (transcribed_text, summarized_text, emotion_detected, interactionid))
        conn.commit()
        cur.close()
        conn.close()
        print(f"[DB] Database updated for Interaction {interactionid}", flush=True)
    except ImportError as e:
        print(f"[DB] Update skipped: {e}", flush=True)
    except Exception as e:
        print(f"[DB ERROR] Could not update database: {e}", flush=True)
