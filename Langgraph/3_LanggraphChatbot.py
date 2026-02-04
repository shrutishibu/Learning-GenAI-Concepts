from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages #reducer function

llm = init_chat_model("google_genai:gemini-2.5-flash-lite", temperature=0)

#Testing API Key
#llm.invoke("How many o's in Google?")

class State(TypedDict):
    #messages: list -> For simple 1 message preservation 
    messages: Annotated[list, add_messages]

def chatbot(state: State) -> State:
    ai_msg = llm.invoke(state["messages"])
    return {"messages": [ai_msg]}
builder = StateGraph(State)

builder.add_node("chatbotNode", chatbot)

builder.add_edge(START, "chatbotNode")
builder.add_edge("chatbotNode", END)

chatGraph = builder.compile()

"""response = chatGraph.invoke({"messages": [HumanMessage(content="What is the capital of India?")]})

response["messages"][-1].content"""

state = None
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting chat. Goodbye!")
        break
    if state is None:
        state: State = {
            "messages": [HumanMessage(content=user_input)]
        }
    else:
        state["messages"].append(HumanMessage(content=user_input))

    state = chatGraph.invoke(state)
    print("Bot:", state["messages"][-1].content)