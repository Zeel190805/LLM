from langchain_groq import ChatGroq
from langchain.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.9)

user_input = "Hi, I'm Kalind. I was born on 22nd December 1999 in Jamnagar."

def summarize_mode():
    messages = [
        SystemMessage(content="You are a summarizer. Summarize the user's input in one concise sentence."),
        HumanMessage(content=user_input),
    ]
    response = llm.invoke(messages)
    return response.content

def professional_mode():
    messages = [
        SystemMessage(content="You are a professional assistant. Respond in a formal, business-like manner."),
        HumanMessage(content=user_input),
    ]
    response = llm.invoke(messages)
    return response.content

def sarcastic_mode():
    messages = [
        SystemMessage(content="You are a sarcastic assistant. Respond with witty sarcasm."),
        HumanMessage(content=user_input),
    ]
    response = llm.invoke(messages)
    return response.content

if __name__ == "__main__":
    print("=" * 60)
    print("SUMMARIZE MODE")
    print("=" * 60)
    print(summarize_mode())
    print("\n")
    
    print("=" * 60)
    print("PROFESSIONAL MODE")
    print("=" * 60)
    print(professional_mode())
    print("\n")
    
    print("=" * 60)
    print("SARCASTIC MODE")
    print("=" * 60)
    print(sarcastic_mode())
