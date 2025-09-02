import chainlit as cl
import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner

# Load env
load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Step 1: Provider
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta",  # FIXED
)

# Step 2: Model
model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",   # FIXED
    openai_client=provider
)

# Step 3: Config
run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

# Step 4: Agent
math_Agent = Agent(
    name="Helpful Assistant",
    instructions="You are a helpful assistant"
)

# Chainlit Handlers
@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! I'm the Support Agent. How can I help you?").send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")

    # Add user message
    history.append({"role": "user", "content": message.content})

    # Run agent
    result = await Runner.run(
        math_Agent,
        input=message.content,
        run_config=run_config,
    )

    # Save assistant reply
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)

    # Send back to UI
    await cl.Message(content=result.final_output or "⚠️ No response from Gemini.").send()