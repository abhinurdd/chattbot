import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API KEYS ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")

# --- DATABASE ---
DATABASE_FILE = "instagram_analysis.json"

# --- API ENDPOINTS & MODELS ---
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
MODEL_NORMALIZER = "openai/gpt-oss-20b:free"
MODEL_SELECTOR = "openai/gpt-oss-20b:free"
MODEL_SCORING = "openai/gpt-oss-20b:free"
