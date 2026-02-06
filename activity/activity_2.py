from langchain_groq import ChatGroq
from langchain.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

user_input = "Tell me small joke about programming."
system_prompt = "You are a creative storyteller. Write engaging and imaginative stories."

def run_with_temperature(temp):
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=temp)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ]
    res = llm.invoke(messages)
    return res.content

if __name__ == "__main__":
    temperatures = [0.0, 0.4, 0.9]
    
    for temp in temperatures:
        print("=" * 60)
        print(f"TEMPERATURE: {temp}")
        print("=" * 60)
        print(run_with_temperature(temp))
        print("\n")
