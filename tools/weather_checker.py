from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv
import os
import re
import json
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENWEATHER_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1:free"
BASE_URL = "http://api.openweathermap.org/data/2.5/forecast"

SYSTEM_PROMPT = """
    You are a helpful assistant. Check the weather forecast and give summary, chance of rain and suggest some advice in response. State each field in one line. Don't give any other information apart from the precaution.

    Input:
    Analyze the weather forecast for Patna, Bihar:
    2025-04-24 00:00:00 | 20.22°C | light rain
    2025-04-24 03:00:00 | 19.99°C | light rain
    2025-04-24 06:00:00 | 19.77°C | light rain

    Output:
    ```json
    {
        "summary": "Light rain is forecasted for the next 3 days.",
        "chance_of_rain": "80%",
        "advice": "Light rain is forecasted. Carry an umbrella.",
        "general_packing_tips": [
            "Wear waterproof clothing.",
            "Bring an umbrella with you.",
            "Bring a rain jacket."
        ],
    }
    ```
"""

class WeatherChecker:
    def __init__(self, city):
        self.city = city

    def get_forecast(self, start_date=None, end_date=None):
        response = requests.get(f"{BASE_URL}?appid={OPENWEATHER_API_KEY}&q={self.city}&units=metric")

        if response.status_code != 200:
            raise Exception(f"Weather API Error: {response.status_code} - {response.text}")

        data = response.json()["list"]

        if start_date and end_date:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            data = [f for f in data if start <= datetime.strptime(f["dt_txt"], "%Y-%m-%d %H:%M:%S") <= end]

        return [
            {
                "time": f["dt_txt"],
                "temp": f["main"]["temp"],
                "desc": f["weather"][0]["description"]
            }
            for f in data
        ]

class WeatherAnalyzer:
    def __init__(self):
        self.model = MODEL
        self.api_key = OPENROUTER_API_KEY
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    def analyze(self, city, forecast_str):
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Analyze the weather forecast for {city}:\n{forecast_str}"
                    }
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "weather",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "summary": {"type": "string"},
                                "chance_of_rain": {"type": "string"},
                                "advice": {"type": "string"},
                                "general_packing_tips": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                            },
                            "required": ["summary", "chance_of_rain", "advice", "general_packing_tips"],
                            "additionalProperties": False,
                        },
                    },
                },
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()

            data = response.json()
            return json.loads(data["choices"][0]["message"]["content"])

        except Exception as e:
            print(f"LLM API Error: {e}")
            return {
                "summary": "Unable to fetch structured data",
                "chance_of_rain": "Unknown",
                "advice": "Check official sources",
                "general_packing_tips": []
            }

    def _extract_json(self, response):
        try:
            match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            json_str = match.group(1) if match else response
            return json.loads(json_str)
        except Exception as e:
            print(f"JSON extraction error: {e}")
            return {
                "summary": "Unable to parse weather data",
                "chance_of_rain": "Unknown",
                "advice": "Check official forecasts",
                "general_packing_tips": []
            }

class WeatherPlanner:
    def __init__(self, city, start_date, end_date):
        self.city = city
        self.start_date = start_date
        self.end_date = end_date
        self.weather_checker = WeatherChecker(city)
        self.weather_analyzer = WeatherAnalyzer()

    def _get_date_range(self):
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end - start).days + 2)]

    def plan_trip(self):
        forecast = self.weather_checker.get_forecast(self.start_date, self.end_date)
        trip_dates = self._get_date_range()

        result = {
            "location": self.city,
            "trip_dates": trip_dates,
            "daily_weather": {},
            "general_packing_tips": None
        }

        for date in trip_dates:
            result["daily_weather"][date] = {}

            daily = [f for f in forecast if date in f["time"]]
            if not daily:
                result["daily_weather"][date] = {
                    "summary": "No forecast data available",
                    "chance_of_rain": "Unknown",
                    "advice": "Check latest updates before planning outdoor activities",
                    "general_packing_tips": []
                }
                continue

            forecast_str = "\n".join(f'{f["time"]} | {f["temp"]}°C | {f["desc"]}' for f in daily)
            analysis = self.weather_analyzer.analyze(self.city, forecast_str)

            result["daily_weather"][date] = {
                "summary": analysis.get("summary", "N/A"),
                "chance_of_rain": analysis.get("chance_of_rain", "Unknown"),
                "advice": analysis.get("advice", "Stay alert"),
                "general_packing_tips": analysis.get("general_packing_tips", [])
            }

        return result

if __name__ == "__main__":
    planner = WeatherPlanner(city="New Delhi", start_date="2025-04-24", end_date="2025-04-26")
    weather_plan = planner.plan_trip()

    print(json.dumps(weather_plan, indent=4))
    
    with open('final_response.json', 'w') as f:
        json.dump(weather_plan, f, indent=4)