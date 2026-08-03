# 🧠 AG-OS — Cognitive Memory Assistant

An AI-powered assistant that helps individuals with short-term memory loss using **face recognition**, **audio transcription**, and **LLM-powered summarization**.

---

## ⚡ Quick Start (Windows — Easiest Way)

> **Requirements:** Python 3.11+, Node.js 18+, PostgreSQL (or a free [Neon](https://neon.tech) cloud DB)

### Step 1 — Clone the repo
```bash
git clone https://github.com/Vishal17082k06/DBMS_proj_backend.git
cd DBMS_proj_backend
```

### Step 2 — Set up your environment file
```bash
copy .env.example .env
```
Then open `.env` and fill in:
- `DB_PASSWORD` — your PostgreSQL password (or use a free [Neon](https://neon.tech) DB)
- `GROQ_API_KEY` — free key from [console.groq.com](https://console.groq.com) *(no credit card needed)*

### Step 3 — Run the setup script
```bash
setup.bat
```
This will:
- Create a Python virtual environment (`venv/`)
- Install all Python packages from `requirements.txt`
- Install frontend npm packages

### Step 4 — Start the app
```bash
start.bat
```
Opens two terminal windows:
- **Backend API** → http://localhost:8004
- **Frontend UI** → http://localhost:5173

Open **http://localhost:5173** in your browser and you're live! 🎉

---

## 🐳 Docker Quick Start (Cross-platform)

> **Requirements:** [Docker Desktop](https://www.docker.com/products/docker-desktop/)

```bash
# Copy env file and fill in your credentials
copy .env.example .env

# Start everything
docker compose up --build
```

- **Backend** → http://localhost:8004
- **Frontend** → http://localhost:5173

---

## 🔑 Credentials Guide

| What you need | Where to get it | Cost |
|---|---|---|
| **Groq API Key** | [console.groq.com](https://console.groq.com) | 🆓 Free |
| **PostgreSQL DB** | [neon.tech](https://neon.tech) | 🆓 Free tier |
| **OpenAI API Key** | [platform.openai.com](https://platform.openai.com) | 💳 Paid (optional) |
| **Google Calendar** | [console.cloud.google.com](https://console.cloud.google.com) | 🆓 Free (optional) |

> **Groq + Neon = 100% free setup** ✅

---

## 🤖 Switching LLM Providers

Use the **LLM PROVIDER** panel in the bottom-right corner of the UI to switch between:

| Provider | Model | Speed | Cost |
|---|---|---|---|
| **Groq** *(default)* | llama-3.1-8b-instant | ⚡ Very fast | 🆓 Free |
| **OpenAI** | gpt-4o-mini | 🐢 Moderate | 💳 Paid |
| **Ollama** | llama3 (local) | 🔒 Private | 🆓 Free |

The switch takes effect immediately — no restart needed.

---

## 🏗️ Project Structure

```
DBMS_proj_backend/
├── app/
│   ├── services/
│   │   ├── face_recognition/     ← Main FastAPI app (port 8004)
│   │   ├── voice_app/            ← Audio transcription (Whisper)
│   │   ├── reminder_app/         ← Google Calendar sync
│   │   └── conversation_summarizer.py  ← LLM summarization
│   ├── ai_models/
│   │   └── interaction/          ← Face + LLM pipeline
│   └── database/                 ← DB connection + queries
├── frontend/                     ← React/Vite UI
├── .env.example                  ← Environment template
├── requirements.txt              ← Python dependencies
├── setup.bat                     ← One-click setup (Windows)
├── start.bat                     ← One-click start (Windows)
└── docker-compose.yml            ← Docker setup (all platforms)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | FastAPI, Python 3.11 |
| **Database** | PostgreSQL (via psycopg2) |
| **Face Recognition** | DeepFace, FaceNet, OpenCV, YOLOv8 |
| **Speech-to-Text** | OpenAI Whisper / faster-whisper |
| **LLM** | Groq (llama-3.1), OpenAI GPT-4o, Ollama |
| **Frontend** | React + Vite |
| **Scheduler** | APScheduler |
| **Calendar** | Google Calendar API |

---

## 📋 Manual Setup (if scripts don't work)

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start backend
venv\Scripts\python -m uvicorn app.services.face_recognition.main:app --reload --port 8004

# In a new terminal — start frontend
cd frontend
npm install
npm run dev
```

---

## 📄 API Docs

Once the backend is running, visit:
- **Swagger UI**: http://localhost:8004/docs
- **ReDoc**: http://localhost:8004/redoc

---

## License

Proprietary — Cognitive Healthcare DBMS Project
