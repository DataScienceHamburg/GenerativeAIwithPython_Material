# %% packages
from agents import Agent, Runner, input_guardrail, \
RunContextWrapper, GuardrailFunctionOutput, InputGuardrailTripwireTriggered
import asyncio
from pydantic import BaseModel
 
# %% guardrail check
class PizzaHotlineGuardrail(BaseModel):
    is_not_pizza_related: bool
    reason: str
 
# %% agents
guardrail_agent = Agent(
    name="guardrail_agent",
    instructions="""
    You are a guardrail agent. You check if the user's
request is related to pizza orders.
    """,
    output_type=PizzaHotlineGuardrail)
 
@input_guardrail
async def pizza_hotline_guardrail(
    context: RunContextWrapper,
    agent: Agent, 
    input: str | list
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=context)
    return GuardrailFunctionOutput(
        output_info=result,
        tripwire_triggered=result.final_output.is_not_pizza_related
    )
 
dispatcher_agent = Agent(
    name="dispatcher",
    instructions="""
    You are a dispatcher. You determine which agent
should handle the user's request.
    You only accept requests related to pizza orders.
    """,
    input_guardrails=[pizza_hotline_guardrail]
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