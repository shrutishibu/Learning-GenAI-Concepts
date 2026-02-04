from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY Not found in .env")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite"
)

response = llm.invoke("How many O's are in Google? How did you verify your answer?")
#reasoning_tokens = response.usage_metadata["output_token_details"]["reasoning"]

print("Response:", response.content)