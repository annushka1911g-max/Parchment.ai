import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Read API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Verify API Key exists
if GOOGLE_API_KEY is None:
    raise ValueError(
        "GOOGLE_API_KEY not found. Please check your .env file."
    )