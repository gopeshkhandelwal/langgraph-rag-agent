import requests
from langchain.tools import tool
from config import get_openweather_api_key

@tool
def CityWeather(city: str) -> str:
    """Fetch current weather for a given city using the OpenWeather API."""
    try:
        api_key = get_openweather_api_key()
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        res = requests.get(url).json()
        if "weather" not in res or "main" not in res:
            return f"Could not retrieve weather for '{city}'."
        desc = res['weather'][0]['description']
        temp = res['main']['temp']
        return f"The weather in {city.title()} is {desc} with a temperature of {temp:.1f}°C."
    except Exception as e:
        return f"Couldn't fetch weather: {e}"
