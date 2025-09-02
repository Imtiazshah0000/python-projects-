import chainlit as cl
import os
import logging
import requests
from dotenv import load_dotenv, find_dotenv

try:
    from agents import (
        Agent,
        RunConfig,
        AsyncOpenAI,
        OpenAIChatCompletionsModel,
        Runner,
        function_tool,
        ModelSettings,   
    )
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False

logging.basicConfig(
    filename="web_search",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables
load_dotenv(find_dotenv())
tavily_api_key = os.getenv("TAVILY_API_KEY")


async def tavily_search_fallback(query: str, max_results: int = 5) -> str:
    """Fallback for Tavily search if agents library is unavailable."""
    logging.info(f"Performing Tavily search for query: {query}")
    try:
        headers = {"Content-Type": "application/json"}
        payload = {
            "api_key": tavily_api_key,
            "query": query,
            "max_results": max_results,
        }
        response = requests.post("https://api.tavily.com/search", json=payload, headers=headers)
        response.raise_for_status()
        results = response.json().get("results", [])
        formatted_results = [
            f"{idx}. {result.get('title', 'No title')}\n"
            f"   URL: {result.get('url', 'No URL')}\n"
            f"   Snippet: {result.get('content', 'No content')[:200]}..."
            for idx, result in enumerate(results[:max_results], 1)
        ]
        return "\n".join(formatted_results) if formatted_results else "No results found."
    except Exception as e:
        logging.error(f"Error during Tavily search for query '{query}': {str(e)}")
        return f"Sorry, I couldn't perform the search for '{query}'. Please try again later."


# 🔹 Agents-based setup
if AGENTS_AVAILABLE:
    provider = AsyncOpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    )

    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=provider,
    )

    run_config = RunConfig(
        model=model,
        model_provider=provider,
        tracing_disabled=True,
        model_settings=ModelSettings(
            temperature=0.7,
            max_tokens=1000,
            tool_choice="auto",
        ),
    )

    @function_tool
    async def tavily_search(query: str, max_results: int = 5):
        """Perform a web search using the Tavily API."""
        logging.info(f"Performing Tavily search for query: {query}")
        try:
            headers = {"Content-Type": "application/json"}
            payload = {
                "api_key": tavily_api_key,
                "query": query,
                "max_results": max_results,
            }
            response = requests.post("https://api.tavily.com/search", json=payload, headers=headers)
            response.raise_for_status()
            results = response.json().get("results", [])
            formatted_results = [
                f"{idx}. {result.get('title', 'No title')}\n"
                f"   URL: {result.get('url', 'No URL')}\n"
                f"   Snippet: {result.get('content', 'No content')[:200]}..."
                for idx, result in enumerate(results[:max_results], 1)
            ]
            return "\n".join(formatted_results) if formatted_results else "No results found."
        except Exception as e:
            raise Exception(f"Tavily API error: {str(e)}")

    search_agent = Agent(
        instructions=(
            "You are a web search assistant. "
            "Use tavily_search to fetch results, summarize concisely, and respond politely."
        ),
        name="WebSearchAgent",
        tools=[tavily_search],
    )

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="👋 Hello! I'm your Web Search Assistant. Ask me anything!").send()


@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})

    if AGENTS_AVAILABLE:
        result = await Runner.run(search_agent, input=history, run_config=run_config)
        response = result.final_output
        logging.info(f"SearchAgent response: {response}")
    else:
        response = await tavily_search_fallback(message.content)
        logging.info(f"Fallback response: {response}")

    history.append({"role": "assistant", "content": response})
    cl.user_session.set("history", history)
    await cl.Message(content=response).send()