import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent

# Path to the FAQ JSON dataset
DATASET_PATH = os.getenv("DATASET_PATH", str(BASE_DIR / "data" / "university_faq.json"))

# ChromaDB storage directory
DB_DIR = os.getenv("DB_DIR", str(BASE_DIR / "chroma_db"))

# ChromaDB collection name
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "faq_collection")

# Groq API Configuration (Cloud Inference)
def get_groq_api_key() -> str:
    load_dotenv(override=True)
    return os.getenv("GROQ_API_KEY", "")

GROQ_API_KEY = get_groq_api_key()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
LLM_PROVIDER = "groq"

# Domain name for the chatbot (e.g., "University", "Motor Company")
# If not set, we will try to infer it from the dataset filename
CHATBOT_DOMAIN = os.getenv("CHATBOT_DOMAIN", "")

def get_chatbot_domain() -> str:
    if CHATBOT_DOMAIN:
        return CHATBOT_DOMAIN
    
    # Simple inference from dataset filename
    filename = Path(DATASET_PATH).name.lower()
    if "rivermount" in filename:
        return "Rivermount University"
    elif "university" in filename:
        return "University"
    elif "motor" in filename or "bike" in filename:
        return "Motor Company"
    elif "hospital" in filename or "medical" in filename:
        return "Hospital/Healthcare"
    elif "it" in filename or "tech" in filename:
        return "IT/Technology Company"
    else:
        return "Support Desk"

def get_welcome_message() -> str:
    domain = get_chatbot_domain()
    return f"Welcome to the {domain} Chatbot! 👋 What can I help you with today?"

def get_system_prompt() -> str:
    domain = get_chatbot_domain()
    return (
        f"You are the helpful virtual assistant for {domain}.\n"
        "Rules:\n"
        "1. For greetings or pleasantries, respond warmly and concisely.\n"
        "2. For general concepts (e.g. GPA definition), provide a short clear explanation.\n"
        f"3. For questions about {domain}, answer directly and accurately using the context below.\n\n"
        "Context:\n{{context}}"
    )
