import chainlit as cl
import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, function_tool, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, ModelSettings

# Configure logging
logging.basicConfig(
    filename='hotel_assistant.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")  # Use OpenAI API key

# Simulated hotel database
HOTEL_DATABASE = {
    "sunshine": {
        "name": "Hotel Sunshine",
        "location": "Miami, FL",
        "amenities": ["Pool", "Spa", "Free Wi-Fi"],
        "availability": {"2025-09-01": 10, "2025-09-02": 8},
        "price_per_night": 150
    },
    "moonlight": {
        "name": "Hotel Moonlight",
        "location": "Seattle, WA",
        "amenities": ["Gym", "Restaurant", "Pet-Friendly"],
        "availability": {"2025-09-01": 5, "2025-09-02": 3},
        "price_per_night": 120
    },
    "starlight": {
        "name": "Hotel Starlight",
        "location": "New York, NY",
        "amenities": ["Bar", "Conference Room", "Free Breakfast"],
        "availability": {"2025-09-01": 15, "2025-09-02": 12},
        "price_per_night": 200
    }
}

# Step 1: Provider and Model Setup
provider = AsyncOpenAI(
    api_key=openai_api_key,
    base_url="https://api.openai.com/v1"
)
model = OpenAIChatCompletionsModel(
    model="gpt-4o-mini",
    openai_client=provider
)
run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
    model_settings=ModelSettings(
        temperature=0.7,
        max_tokens=1000,
        tool_choice="auto",
        metadata={"app": "HotelAssistant", "version": "1.0"}
    )
)

# Step 2: Function Tool
@function_tool
async def get_hotel_details(hotel_name: str, date: str = None):
    """Fetch details for a specified hotel, optionally for a specific date."""
    async def is_enabled(context: Any) -> bool:
        return any(keyword in str(context).lower() for keyword in ["hotel", "price", "availability", "amenities", "location"])

    async def error_function(error: Exception) -> str:
        logging.error(f"Error fetching hotel details for {hotel_name}: {str(error)}")
        return f"Sorry, I couldn't find hotel '{hotel_name}'. Please try Sunshine, Moonlight, or Starlight."

    logging.info(f"Fetching details for hotel: {hotel_name}, date: {date}")
    hotel_key = next((key for key in HOTEL_DATABASE if key in hotel_name.lower()), None)
    if hotel_key:
        hotel = HOTEL_DATABASE[hotel_key]
        details = (
            f"🏨 Hotel: {hotel['name']}\n"
            f"📍 Location: {hotel['location']}\n"
            f"✨ Amenities: {', '.join(hotel['amenities'])}\n"
            f"💵 Price per night: ${hotel['price_per_night']}"
        )
        if date and date in hotel['availability']:
            details += f"\n🗓️ Availability on {date}: {hotel['availability'][date]} rooms"
        elif date:
            details += f"\n⚠️ No availability data for {date}"
        return details
    raise ValueError(f"Hotel {hotel_name} not found")

# Step 3: Dynamic Instructions
def generate_dynamic_instructions(context: Dict, agent: Agent) -> str:
    """Generate dynamic instructions based on user query and context."""
    query = context.get("query", "").lower()
    history = context.get("history", [])
    hotel_name = None

    # Check query for hotel names
    for key in HOTEL_DATABASE:
        if key in query:
            hotel_name = HOTEL_DATABASE[key]["name"]
            break

    # Check history for hotel context
    if not hotel_name:
        for msg in reversed(history):
            if msg["role"] == "user":
                for key in HOTEL_DATABASE:
                    if key in msg["content"].lower():
                        hotel_name = HOTEL_DATABASE[key]["name"]
                        break
                if hotel_name:
                    break

    if hotel_name:
        instructions = f"""You are a helpful hotel assistant for {hotel_name}. 
        Provide information or assist with bookings for this hotel using the get_hotel_details tool."""
    else:
        instructions = """You are a hotel assistant managing multiple hotels 
        (Sunshine, Moonlight, Starlight). Identify the hotel from the query or context 
        and use get_hotel_details to provide information. 
        If no hotel is specified, ask the user to clarify."""

    logging.info(f"Generated instructions: {instructions}")
    return instructions

# Step 4: Agent
hotel_agent = Agent(
    instructions=generate_dynamic_instructions,
    name="HotelAssistant",
    tools=[get_hotel_details]
)

# Step 5: Chainlit Handlers
@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(
        content="👋 Hello! I'm your Hotel Assistant for Sunshine, Moonlight, and Starlight.\n"
                "You can ask about hotel details or availability.\n"
                "Example: *Sunshine availability on 2025-09-01*\n"
                "Or: *Hotel Sunshine ka location kya hai?*"
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history", [])
    history.append({"role": "user", "content": message.content})

    try:
        context = {"query": message.content, "history": history}
        result = await Runner.run(
            hotel_agent,
            input=[{"role": "user", "content": message.content}],
            context=context,
            run_config=run_config
        )

        response = result.final_output if result.final_output else "Sorry, I couldn’t find any info. Try asking about Sunshine, Moonlight, or Starlight."
        
        if isinstance(response, list):
            response = " ".join([str(x) for x in response])

        logging.info(f"Agent response: {response}")

    except Exception as e:
        logging.error(f"Error processing query '{message.content}': {str(e)}")
        response = f"⚠️ Sorry, I couldn't process your request: {str(e)}"

    history.append({"role": "assistant", "content": response})
    cl.user_session.set("history", history)
    await cl.Message(content=response).send()

# Step 6: Console Test Function
async def test_hotel_assistant():
    queries = [
        "Hotel Sunshine ka location kya hai?",
        "Check availability at Moonlight on 2025-09-01",
        "What’s the price per night at Starlight?",
        "Tell me about Hotel Sunshine"
    ]
    
    for query in queries:
        history = cl.user_session.get("history", [])
        history.append({"role": "user", "content": query})
        cl.user_session.set("history", history)
        
        logging.info(f"Testing query: {query}")
        try:
            context = {"query": query, "history": history}
            result = await Runner.run(
                hotel_agent,
                input=[{"role": "user", "content": query}],
                context=context,
                run_config=run_config
            )
            response = result.final_output if result.final_output else "Sorry, I couldn’t find any info."
            if isinstance(response, list):
                response = " ".join([str(x) for x in response])
            print(f"Question: {query}")
            print(f"Answer: {response}\n")
            history.append({"role": "assistant", "content": response})
            cl.user_session.set("history", history)
        except Exception as e:
            print(f"Question: {query}")
            print(f"Answer: Error - {str(e)}\n")

# Run tests
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_hotel_assistant())