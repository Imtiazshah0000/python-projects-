import chainlit as cl
import os
import logging
from enum import Enum
from pydantic import BaseModel
from typing import List, Union
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, input_guardrail, output_guardrail, GuardrailFunctionOutput, RunContextWrapper, TResponseInputItem, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig,ModelSettings

# Configure logging
logging.basicConfig(
    filename='guardrails.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Step 1: Define Enum Classes for Math Topic and Complexity
class MathTopicType(str, Enum):
    ARITHMETIC = "arithmetic"
    ALGEBRA = "algebra"
    GEOMETRY = "geometry"
    CALCULUS = "calculus"
    STATISTICS = "statistics"
    OTHER = "other"

class MathComplexityLevel(str, Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

# Step 2: Define Input Guardrail Output Model
class MathHomeworkOutput(BaseModel):
    is_math_homework: bool
    reasoning: str
    topic_type: MathTopicType
    complexity_level: MathComplexityLevel
    detected_keywords: List[str]
    is_step_by_step_requested: bool
    allow_response: bool
    explanation: str

# Step 3: Define Output Guardrail Output Model
class PoliticalContentOutput(BaseModel):
    contains_political_content: bool
    reasoning: str
    detected_keywords: List[str]

# Step 4: Provider and Model Setup (Using Gemini API)
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
    model_settings=ModelSettings(   # <-- yahan dict ki jagah object use karo
        temperature=0.7,
        max_tokens=1000,
        tool_choice="none"
    )
)

# Step 5: Input Guardrail Agent
guardrail_agent = Agent(
    name="MathQueryAnalyzer",
    instructions="""You are an expert at detecting and blocking attempts to get math homework help.
    - Identify if the query is a math homework question (e.g., 'Solve 2x + 3 = 11').
    - Detect disguised homework requests (e.g., 'I’m practicing algebra and stuck').
    - Allow conceptual math questions (e.g., 'Why is negative times negative positive?').
    - Return structured output with reasoning and classification.""",
    output_type=MathHomeworkOutput
)

# Step 6: Output Guardrail Agent
political_guardrail_agent = Agent(
    name="PoliticalContentAnalyzer",
    instructions="""You are an expert at detecting political content in text.
    - Check if the text contains political topics (e.g., elections, government, political parties) or references to political figures (e.g., specific names).
    - Return structured output with reasoning and detected keywords.
    - Be strict: flag any mention of political terms or figures.""",
    output_type=PoliticalContentOutput
)

# Step 7: Input Guardrail Logic
@input_guardrail
async def math_guardrail(ctx: RunContextWrapper[None], agent: Agent, input: Union[str, List[TResponseInputItem]]) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)
    output = result.final_output
    tripwire = (
        output.is_math_homework or
        not output.allow_response or
        output.is_step_by_step_requested or
        output.complexity_level != "basic" or
        any(kw in str(input).lower() for kw in ["solve", "solution", "answer", "help with", "step", "explain how", "calculate", "find", "determine", "evaluate", "work out"])
    )
    if tripwire:
        logging.warning(f"Input guardrail triggered for input: {input}. Reasoning: {output.reasoning}")
    return GuardrailFunctionOutput(output_info=output, tripwire_triggered=tripwire)

# Step 8: Output Guardrail Logic
@output_guardrail
async def political_guardrail(ctx: RunContextWrapper, agent: Agent, output: str) -> GuardrailFunctionOutput:
    political_keywords = ["election", "government", "politics", "political", "party", "democrat", "republican", "trump", "biden", "president", "congress", "senate", "policy"]
    result = await Runner.run(political_guardrail_agent, output, context=ctx.context)
    output_info = result.final_output
    detected_keywords = [kw for kw in political_keywords if kw in output.lower()]
    tripwire = output_info.contains_political_content or len(detected_keywords) > 0
    if tripwire:
        logging.warning(f"Output guardrail triggered for output: {output}. Reasoning: {output_info.reasoning}. Detected keywords: {detected_keywords}")
    return GuardrailFunctionOutput(
        output_info=output_info,
        tripwire_triggered=tripwire
    )

# Step 9: Educational Support Agent
agent = Agent(
    name="EducationalSupportAssistant",
    instructions="""You are an educational support assistant focused on promoting genuine math learning.
    - Provide conceptual explanations for math topics.
    - Avoid giving direct answers to homework questions.
    - Respond politely and encourage learning.""",
    input_guardrails=[math_guardrail],
    output_guardrails=[political_guardrail]
)

# Step 10: Chainlit Handlers
@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! I'm your Math Learning Assistant. Ask me about math concepts!").send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})
    
    try:
        result = await Runner.run(agent, input=history, run_config=run_config)
        response = result.final_output
        logging.info(f"Agent response: {response}")
    except Exception as e:
        logging.error(f"Error processing query '{message.content}': {str(e)}")
        response = f"Sorry, I couldn't process your request: {str(e)}"
    
    history.append({"role": "assistant", "content": response})
    cl.user_session.set("history", history)
    await cl.Message(content=response).send()