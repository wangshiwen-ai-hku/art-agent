from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_swarm, create_handoff_tool

from langgraph.graph.state import CompiledStateGraph



def create_swam_agent_and_handoff_tool(agent_name: str, model: str, agents: list[CompiledStateGraph], prompt: str):

    handoff_tools = []
    for agent in agents:
        handoff_tools.append(create_handoff_tool(
            agent_name=agent,
            description=f"Transfer user to the {agent}.",
        ))

    agent = create_react_agent(
        model=model,
        tools=handoff_tools,
        prompt=prompt,
        name=agent_name,
    )

    return agent
# transfer_to_hotel_assistant = create_handoff_tool(
#     agent_name="hotel_assistant",
#     description="Transfer user to the hotel-booking assistant.",
# )
# transfer_to_flight_assistant = create_handoff_tool(
#     agent_name="flight_assistant",
#     description="Transfer user to the flight-booking assistant.",
# )

# flight_assistant = create_react_agent(
#     model="anthropic:claude-3-5-sonnet-latest",
#     tools=[book_flight, transfer_to_hotel_assistant],
#     prompt="You are a flight booking assistant",
#     name="flight_assistant"
# )
# hotel_assistant = create_react_agent(
#     model="anthropic:claude-3-5-sonnet-latest",
#     tools=[book_hotel, transfer_to_flight_assistant],
#     prompt="You are a hotel booking assistant",
#     name="hotel_assistant"
# )

# swarm = create_swarm(
#     agents=[flight_assistant, hotel_assistant],
#     default_active_agent="flight_assistant"
# ).compile()

# for chunk in swarm.stream(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "book a flight from BOS to JFK and a stay at McKittrick Hotel"
#             }
#         ]
#     }
# ):
#     print(chunk)
#     print("\n")