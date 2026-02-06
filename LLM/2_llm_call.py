# Import ChatOllama to interface with a local or remote Ollama LLM
from langchain_ollama import ChatOllama

from langchain_groq import ChatGroq

# Import structured message classes from LangChain
# - SystemMessage: defines instructions for the assistant's behavior
# - HumanMessage: represents input from the user
from langchain.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Step 1: Initialize the LLM model
# -----------------------------

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.9,
)

# llm = ChatOllama(
#     model="llama3.2",  # The LLM model to use (must be running on the Ollama server)
#     base_url="http://localhost:11434",  # The base URL of the Ollama API (local or remote)
#     temperature=0.1,  # Low temperature = deterministic & consistent responses
# )

# -----------------------------
# Step 2: Define conversation messages using structured schema
# -----------------------------

# Instead of using dictionaries for messages, we use LangChain's built-in message types:
# - SystemMessage: for setting instructions or context
# - HumanMessage: for sending user prompts

messages = [
    SystemMessage(
        content="You are a helpful assistant. Convert the user input from English to French. Return JSON"
        # This instructs the model to:
        # 1. Translate English to French
        # 2. Format the result as JSON
    ),
    HumanMessage(
        content="My name is Kalind. I am from Jamnagar in Gujarat. I was born on 22nd December, 1999."
        # This is the English input that the assistant will process
    ),
]

# -----------------------------
# Step 3: Invoke the model and get the response
# -----------------------------

# The 'invoke' method sends the structured messages to the LLM and returns a response
response = llm.invoke(messages)

# -----------------------------
# Step 4: Display the output
# -----------------------------

# Access and print only the assistant's content (the actual response text)
print("Assistant response : " + response.content)
