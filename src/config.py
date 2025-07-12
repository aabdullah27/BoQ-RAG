import os
from dotenv import load_dotenv

load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Set environment variables for LlamaIndex
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY or ""
os.environ["GROQ_API_KEY"] = GROQ_API_KEY or ""

# Qdrant settings
COLLECTION_NAME = "material_boq"
EMBED_MODEL_NAME = "text-embedding-3-small"
VECTOR_SIZE = 1536  # For OpenAI embedding model text-embedding-3-small

# Data directory
DATA_DIR = "data" 