from langchain_groq import ChatGroq
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

@tool
def convert_to_uppercase(text: str) -> str:
    """
    Convert any text to uppercase using Groq API.
    
    Args:
        text: The text you want to convert to uppercase
        
    Returns:
        The text converted to uppercase
    """
    prompt = f"""Convert the following text to uppercase. 
Return ONLY the uppercase text, nothing else.

Text: {text}"""
    
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    response = llm.invoke(messages)
    return response.content.strip()


if __name__ == "__main__":
    # Test the tool
    test_text = input("Enter text to convert to uppercase: ")
    result = convert_to_uppercase.invoke({"text": test_text})
    print(f"\nResult: {result}")
