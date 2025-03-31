import getpass
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDv5E0ApDmQ2YLAzCEEIKGxhR5G7DdyIY4"

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
   
    # other params...
)

# # Comment out the below to opt-out of using LangSmith in this notebook. Not required.
# if not os.environ.get("LANGSMITH_API_KEY"):
#     os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
#     os.environ["LANGSMITH_TRACING"] = "true"

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg)