import os
from dotenv import load_dotenv

# Build an absolute path to the .env file to ensure it's always found.
project_root = os.path.join(os.path.dirname(__file__), '..')
dotenv_path = os.path.join(project_root, '.env')

# Explicitly load the .env file from the defined path.
load_dotenv(dotenv_path=dotenv_path)

# Get the OpenAI API key from the environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# You can add other configurations here later