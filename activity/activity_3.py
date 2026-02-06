from langchain_groq import ChatGroq
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

system_message = SystemMessage(content="You are a helpful assistant. Keep your responses concise and friendly.")
messages = [system_message]
user_message_count = 0

def chat(user_input):
    global messages, user_message_count
    
    messages.append(HumanMessage(content=user_input))
    user_message_count += 1
    
    response = llm.invoke(messages)
    messages.append(AIMessage(content=response.content))
    
    if user_message_count >= 5:
        print("\n⚠️ Chat history is now too long. Resetting memory.\n")
        messages = [system_message]
        user_message_count = 0
    
    return response.content

if __name__ == "__main__":
    print("Chat with the assistant (type 'quit' to exit)\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        response = chat(user_input)
        print(f"Assistant: {response}\n")
