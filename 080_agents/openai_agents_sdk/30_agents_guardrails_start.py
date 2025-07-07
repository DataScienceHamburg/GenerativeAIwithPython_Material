# %% packages
from agents import Agent, Runner, input_guardrail, RunContextWrapper, GuardrailFunctionOutput, InputGuardrailTripwireTriggered
import asyncio
from pydantic import BaseModel
 
# %% guardrail check
# TODO: Pydantic model for guardrail check

# %% agents
# TODO: Agent for guardrail check

#%% input guardrail
# TODO: Input guardrail function

#%% dispatcher agent 
dispatcher_agent = Agent(
    name="dispatcher",
    instructions="""
    You are a dispatcher. You determine which agent
should handle the user's request.
    You only accept requests related to pizza orders.
    """,
    # TODO: Add input guardrail
)

async def main():
    try:
        user_input = "I want to buy a pizza."
        await Runner.run(dispatcher_agent, user_input)
        print(f"User input: {user_input}")
        print("Guardrail did not trip - as expected for pizza request.") 
    except InputGuardrailTripwireTriggered as e:
        print("Tripwire triggered - meaning it was not related to pizza!")
            

if __name__ == "__main__":
    asyncio.run(main())