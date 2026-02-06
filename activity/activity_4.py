from langchain_groq import ChatGroq
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

system_message = SystemMessage(content="You are a helpful assistant. Keep your responses concise and friendly.")
messages = [system_message]

def save_chat_log():
    with open("chat_log.txt", "w", encoding="utf-8") as f:
        f.write(f"Chat Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        for msg in messages[1:]:
            if isinstance(msg, HumanMessage):
                f.write(f"User: {msg.content}\n")
            elif isinstance(msg, AIMessage):
                f.write(f"Assistant: {msg.content}\n\n")

def chat(user_input):
    messages.append(HumanMessage(content=user_input))
    response = llm.invoke(messages)
    messages.append(AIMessage(content=response.content))
    return response.content

if __name__ == "__main__":
    print("Chat with the assistant (type 'bye' to save and exit)\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'bye':
            save_chat_log()
            print("\n✅ Chat saved to chat_log.txt. Goodbye!")
            break
        
        response = chat(user_input)
        print(f"Assistant: {response}\n")
