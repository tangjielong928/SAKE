import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # LLM Configuration
    API_KEY = os.getenv("LLM_API_KEY")
    BASE_URL = os.getenv("LLM_BASE_URL")
    MODEL_NAME = os.getenv("MODEL_NAME") 
    
    # Search Tool Configuration
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")
    
    # Output Configuration
    OUTPUT_DIR = "output_trajectories"
    
    @staticmethod
    def validate():
        if not Config.API_KEY:
            raise ValueError("Missing LLM_API_KEY in .env")
        if not Config.SERPER_API_KEY:
            print("Warning: SERPER_API_KEY not found. Search will be mocked.")