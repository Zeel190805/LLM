from langchain.tools import tool
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()


# -----------------------------
# Step 1: Initialize the LLM
# -----------------------------
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.9)

@tool
def to_upper_case(text: str) -> str:
    """Convert the input text to uppercase."""
    return text.upper()

@tool
def count_words(text: str) -> int:
    """Count the number of words in the input text."""
    return len(text.split())

tools_list = [to_upper_case, count_words]

agent_prompt = """
when user mention upper case or count the words then use resepective functions otherwise give your best answer and nicely behave.
"""

agent = create_agent(
    model=llm,
    tools=tools_list,
    system_prompt=agent_prompt,
)


def get_response_from_agent(user_input):
    input_messages = [
        {"role": "user", "content": user_input},
    ]

    for step in agent.stream({"messages": input_messages}, stream_mode="values"):
        step["messages"][-1].pretty_print()

    print(
        "---------------------------------------------------------------------------------"
    )


get_response_from_agent("Convert 'hELLo world i am Zeel Barvaliya' to uppercase.")


get_response_from_agent(
    "Count the number of words in the following text: 'Hello world i am King Kon, this is a test.'"
)

get_response_from_agent("Convert 'hELLo world i am Zeel Barvaliya' to uppercase.")
