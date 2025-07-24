#%% packages
import operator
from typing import TypedDict, Annotated, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from dotenv import load_dotenv, find_dotenv
from IPython.display import Image, display
load_dotenv(find_dotenv(usecwd=True))
#%% State
class ConversationState(TypedDict):
    """
    Represents the state of our conversation graph.

    Attributes:
        messages: A list of BaseMessage objects, annotated to append new messages.
        summary: An optional string to store the conversation summary.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    summary: str = "" 

#%% model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

#%% Nodes
def conversation_node(state: ConversationState) -> dict:
    """
    Simulates a step in the conversation.
    Adds a new AI message and a new human message to the state's messages list.
    """
    print(f"--- Entering conversation_node ---")
    current_messages = state.get("messages", [])
    message_count = len(current_messages)

    ai_response = llm.invoke([HumanMessage(content=f"AI says: This is response number {message_count // 2 + 1}.")])
    # Simulate Human input
    human_input = HumanMessage(content=f"Human says: My input after AI response {message_count // 2 + 1}.")

    print(f"Current message count: {message_count}")
    print(f"Adding AI message: '{ai_response.content}'")
    print(f"Adding Human message: '{human_input.content}'")

    # Return updates to the state. LangGraph will apply these.
    return {"messages": [ai_response, human_input]}

def summarize_node(state: ConversationState) -> dict:
    """
    Simulates summarizing the conversation.
    Concatenates all messages into a summary string.
    """
    print(f"--- Entering summarize_node ---")
    all_messages = state.get("messages", [])
    full_conversation_text = "\n".join([msg.content for msg in all_messages])

    # Simulate a summary
    summary_text = f"Summary of conversation (total {len(all_messages)} messages):\n{full_conversation_text[:200]}..." # Truncate for brevity

    print(f"Generated summary: '{summary_text}'")
    # Return the summary to update the state
    return {"summary": summary_text}

# 3. Define the Conditional Edge Logic
# This function determines the next node based on the current state.
def should_continue(state: ConversationState) -> str:
    """
    Determines whether to continue the conversation or summarize.
    Based on the image, if message count > 6, summarize; otherwise, continue.
    """
    message_count = len(state.get("messages", []))
    print(f"--- Checking should_continue ---")
    print(f"Total messages for condition check: {message_count}")

    if message_count >= 6: # Changed to >= 6 to match the visual flow for summarization after some turns
        print("Condition met: message count >= 6. Transitioning to 'summarize_conversation'.")
        return "summarize_conversation"
    else:
        print("Condition not met: message count < 6. Transitioning back to 'conversation'.")
        return "conversation"

#%% Build the Graph
builder = StateGraph(ConversationState)

# Add nodes to the workflow
builder.add_node("conversation", conversation_node)
builder.add_node("summarize_conversation", summarize_node)

# Set the entry point
builder.set_entry_point("conversation") # Start directly with the conversation node

# Add conditional edges from the conversation node
# This is the core logic for looping or summarizing
builder.add_conditional_edges(
    "conversation",      # Source node
    should_continue,     # Function to determine next node
    {
        "conversation": "conversation",      # If should_continue returns "conversation", loop back
        "summarize_conversation": "summarize_conversation", # If should_continue returns "summarize_conversation", go to summarize
    }
)

# Add a direct edge from summarize_conversation to END
builder.add_edge("summarize_conversation", END)
builder.add_edge("conversation", END)

#%% Compile the graph
graph = builder.compile()

#%% display graph
display(Image(graph.get_graph().draw_mermaid_png()))

#%% Run the Graph (Example Usage)
print("--- Starting LangGraph Flow ---")

# Initial state: a single human message to kick off the conversation
while True:
    user_query = input("Enter your query: ")
    initial_messages = [HumanMessage(content=user_query)]
    final_state = graph.invoke({"messages": initial_messages})
    print(final_state)
    if user_query.lower() == "exit":
        break
    

# %%
