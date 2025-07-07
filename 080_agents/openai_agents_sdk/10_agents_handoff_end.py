from agents import Agent, Runner
import asyncio

order_agent = Agent(
    name="order_agent",
    instructions="""
    You are an order agent. You take the user's order and confirm it. State that you are an order agent.
    """
)

service_agent = Agent(
    name="service_agent",
    instructions="""
    You are a service agent. You answer questions related to the service. State that you are a service agent.
    """
)

dispatcher_agent = Agent(
    name="dispatcher",
    instructions="""
    You are a dispatcher. You determine which agent should handle the user's request.
    """,
    handoffs=[order_agent, service_agent]
)

async def main():
    result = await Runner.run(dispatcher_agent, "I want to order a pizza")
    print(result)

    result = await Runner.run(dispatcher_agent, "I ordered a while ago, but I haven't received my pizza yet. What's the status?")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())