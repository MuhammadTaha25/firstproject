from langchain_google_genai import GoogleGenerativeAI
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')

def initialize_LLM(openai_api_key=OPENAI_API_KEY, gemini_api_key=GOOGLE_API_KEY):
    """
    Initialize a Language Learning Model (LLM) using OpenAI or Gemini based on the availability of API keys.

    Parameters:
        openai_api_key (str, optional): Your OpenAI API key. Defaults to None and uses the environment variable if not provided.
        gemini_api_key (str, optional): Your Gemini API key. Defaults to None and uses the environment variable if not provided.

    Returns:
        object: An instance of ChatOpenAI (OpenAI model) or GoogleGenerativeAI (Gemini model).
    """
    # Use explicitly provided API keys or fallback to environment variables
    openai_api_key = openai_api_key or OPENAI_API_KEY
    gemini_api_key = gemini_api_key or GOOGLE_API_KEY

    if openai_api_key:
        try:
            model_name = "gpt-4o-mini"
            LLM = ChatOpenAI(
                model_name=model_name,
                openai_api_key=openai_api_key,
                temperature=0
            )
            print("Using OpenAI's GPT-4 model.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI model: {e}")
    elif gemini_api_key:
        try:
            model_name = "gemini-1.5-flash-002"
            LLM = GoogleGenerativeAI(
                model=model_name,
                google_api_key=gemini_api_key
            )
            print("Using Gemini's model.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini model: {e}")
    else:
        raise ValueError("No API keys provided. Please set the OpenAI or Gemini API key.")

    return LLM

