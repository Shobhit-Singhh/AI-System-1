import os
from dotenv import load_dotenv

def setup_environment():
    load_dotenv()
    
    openai_api_key = os.getenv("OPENAI_API_KEY")

    
    if not openai_api_key:
        raise ValueError(
            "Missing environment variables. Please check your .env file.")
    
    os.environ["OPENAI_API_KEY"] = openai_api_key