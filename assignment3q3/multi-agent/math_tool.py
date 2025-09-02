import os
from agents import Agent,Runner,AsyncOpenAI,OpenAIChatCompletionsModel,function_tool
from agents.run import RunConfig
from dotenv import load_dotenv

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
   raise ValueError("API_KEY is not set.")

external_api_key = AsyncOpenAI(
   api_key=gemini_api_key,
   base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
   model="gemini-2.5-flash",
   openai_client=external_api_key
)


config = RunConfig(
   model=model,
   tracing_disabled=True,
)

@function_tool
def add(n1,n2):
    print("plus tool fire ------->")
    return f"your answer is {n1+n2}"


def main():
   print("🧮 My Assignment 3")


   agent = Agent(
      name="Math Assistant",
      instructions="you are math Assistant.",
      model=model,
      tools=[add],
   )

   while True:
       user_question = input(" You :) =")
       if user_question.lower() in ["exit", "quit"]:
           print("Goodbye! 👋")
           break

   result = Runner.run_sync(agent, user_question, run_config=config)
   print("🤖 Result :", result.final_output)

if __name__ == "__main__":
   main()
 