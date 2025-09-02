import chainlit as cl
import os
import uuid
import logging
from functools import wraps
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, function_tool, ModelSettings
from dotenv import load_dotenv, find_dotenv

# Configure logging
logging.basicConfig(
    filename='customer_support_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Simulated order database
ORDER_DATABASE = {
    "12345": {"status": "Shipped", "items": ["Laptop", "Mouse"], "date": "2025-08-20"},
    "67890": {"status": "Processing", "items": ["Phone"], "date": "2025-08-22"}
}

# Step 1: Custom Guardrail Decorator (Fallback if agents.guardrail is not available)
def guardrail(func):
    """Custom decorator to check for negative/offensive language in queries."""
    negative_words = ["stupid", "hate", "idiot", "terrible"]
    
    @wraps(func)
    async def wrapper(query: str, *args, **kwargs) -> str:
        """Check query for negative words and return a warning if detected."""
        if any(word in query.lower() for word in negative_words):
            logging.warning(f"Negative language detected in query: {query}")
            return "Please rephrase your query in a positive manner, and I'll be happy to assist!"
        return await func(query, *args, **kwargs)
    return wrapper

# Step 2: Provider
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
)

# Step 3: Model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)


# Step 4: Run Configuration with ModelSettings
run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
    model_settings=ModelSettings(
        tool_choice="auto",   # Let model decide when to use tools
        metadata={"store_id": "STORE123", "customer_support_version": "1.0"},
        temperature=0.2
    )
)

# Step 5: Function Tool for Order Status
@function_tool
async def get_order_status(order_id: str):
    """Fetch the status of an order given its order ID."""
    def is_enabled(query: str) -> bool:
        """Enable tool only for queries containing 'order'."""
        return "order" in query.lower()
    
    async def error_function(error: Exception) -> str:
        """Handle errors gracefully."""
        logging.error(f"Error fetching order status for {order_id}: {str(error)}")
        return f"Sorry, I couldn't find order {order_id}. Please check the ID and try again."

    logging.info(f"Fetching order status for order_id: {order_id}")
    if order_id in ORDER_DATABASE:
        order = ORDER_DATABASE[order_id]
        return f"Order {order_id}: Status - {order['status']}, Items - {', '.join(order['items'])}, Date - {order['date']}"
    else:
        raise ValueError(f"Order {order_id} not found")

# Step 6: Bot Agent for FAQs and Order Lookup
bot_agent = Agent(
    name="CustomerSupportBot",
    instructions="""You are a customer support bot for an e-commerce store. Your tasks are:
    - Answer FAQs:
      Q: What are your store hours? A: We are open 24/7 online!
      Q: What is the return policy? A: Returns are accepted within 30 days with a receipt.
    - Use the get_order_status tool for order-related queries.
    - If the query is complex or contains negative sentiment, escalate to the Human Agent.
    - Be polite and professional in all responses.""",
    tools=[get_order_status]
)

# Step 7: Human Agent for Escalation
human_agent = Agent(
    instructions="""You are a human customer support agent. Handle escalated queries with empathy and professionalism. Provide a generic response indicating a human will follow up, and log the escalation.""",
    name="HumanAgent"
)

# Step 8: Chat Start Handler
@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    cl.user_session.set("customer_id", str(uuid.uuid4()))  # Assign unique customer ID
    await cl.Message(content="Hello! I'm your Customer Support Bot. Ask about store hours, return policy, or order status (e.g., 'Check order 12345').").send()

# Step 9: Message Handler with Guardrails, Tool Usage, and Handoff
@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")
    customer_id = cl.user_session.get("customer_id")
    
    # Apply guardrail to check for negative language
    @guardrail
    async def check_language(query: str) -> str:
        return query  # Return query unchanged if no negative words (handled by decorator)

    cleaned_query = await check_language(message.content)
    if cleaned_query != message.content:
        await cl.Message(content=cleaned_query).send()
        return

    # Update history with user input
    history.append({"role": "user", "content": cleaned_query})
    
    # Check for complex or negative sentiment queries
    complex_keywords = ["problem", "issue", "complaint", "not working"]
    is_complex = any(keyword in cleaned_query.lower() for keyword in complex_keywords)
    
    if is_complex:
        # Escalate to Human Agent
        logging.info(f"Escalating query to HumanAgent: {cleaned_query}")
        result = await Runner.run(
            human_agent,
            input=history,
            run_config=run_config
        )
        response = f"Thank you for reaching out. Your query ('{cleaned_query}') has been escalated to a human agent. Someone will follow up with you soon!"
        logging.info(f"Escalation response: {response}")
    else:
        # Run BotAgent for FAQs or order status
        result = await Runner.run(
            bot_agent,
            input=history,
            run_config=run_config
        )
        response = result.final_output
        logging.info(f"BotAgent response: {response}")

    # Update history with assistant response
    history.append({"role": "assistant", "content": response})
    cl.user_session.set("history", history)
    
    # Send response to user
    await cl.Message(content=response).send()