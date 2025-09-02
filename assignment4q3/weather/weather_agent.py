import os
import requests
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
from agents.run import RunConfig
from dotenv import load_dotenv


load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
weather_api_key = os.getenv("WEATHER_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")
if not weather_api_key:
    raise ValueError("WEATHER_API_KEY environment variable is not set.")


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
def get_weather(location: str) -> str:
    """
    Gets current weather and 3-day forecast using WeatherAPI.
    """
    base_url = "http://api.weatherapi.com/v1/forecast.json"
    params = {
        "key": weather_api_key,
        "q": location,
        "days": 7,
        "aqi": "yes",
        "alerts": "yes"
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        location_info = data["location"]
        current = data["current"]
        forecast_days = data["forecast"]["forecastday"]

        city = location_info["name"]
        country = location_info["country"]
        temp_c = current["temp_c"]
        condition = current["condition"]["text"]

        forecast_summary = "\n".join(
            [
                f"- {day['date']}: {day['day']['condition']['text']}, "
                f"High: {day['day']['maxtemp_c']}°C, Low: {day['day']['mintemp_c']}°C"
                for day in forecast_days
            ]
        )

        return (
            f"📍 Weather for {city}, {country}:\n"
            f"🌡️ Current: {temp_c}°C, {condition}\n"
            f"🔮 Forecast:\n{forecast_summary}"
        )

    except requests.exceptions.RequestException as e:
        return f"Failed to fetch weather data: {e}"
    except KeyError:
        return f"Invalid response from weather service for location: {location}"

def main():
    print("")
    print("🌦️ Welcome to the Weather Tool Bot!")
    print("")
    print("You can ask me about the weather in any city.")
    print("")
    print("Type 'exit' or 'quit' to quit the program.")
    print("")

    agent = Agent(
        name="Weather Tool Bot",
        instructions="You are a weather agent. Use the weather tool to get the weather information.",
        model=model,
        tools=[get_weather],
    )

    while True:
        user_question = input("You :) = ")
        if user_question.lower() in ["exit", "quit"]:
            print("Goodbye! 👋")
            break

        result = Runner.run_sync(agent, user_question, run_config=config)
        print("🌦️ Bot:", result.final_output)

if __name__ == "__main__":
    main()