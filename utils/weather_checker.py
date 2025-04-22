from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv
import os
import json
from typing import List
from pydantic import BaseModel, Field
from instructor import from_groq, Mode
from groq import Groq

load_dotenv()
OPENWEATHER_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
MODEL = "llama-3.3-70b-versatile"
BASE_URL = "http://api.openweathermap.org/data/2.5/forecast"

class WeatherResponse(BaseModel):
    summary: str = Field(alias="summary")
    chance_of_rain: str = Field(alias="chance_of_rain")
    advice: str = Field(alias="advice")
    general_packing_tips: List[str] = Field(alias="packing_tips")

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
        self.client = from_groq(
            Groq(api_key=GROQ_API_KEY),
            mode=Mode.JSON
        )

    def analyze(self, city, forecast_str) -> WeatherResponse:
        prompt = f"""
        You are a helpful assistant. Analyze the weather forecast for {city}. Based on the input, return only a JSON object with the following fields:

        - summary: One-line summary of the weather for the day.
        - chance_of_rain: Estimated chance of rain as a percentage.
        - advice: One line of travel advice based on the weather.
        - general_packing_tips: A short list of 2–4 things to pack based on the weather (like umbrella, light clothes, etc.).

        Input Forecast:
        {forecast_str}
        """

        response = self.client.chat.completions.create(
            model=MODEL,
            response_model=WeatherResponse,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that strictly outputs structured JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5
        )

        return response

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
            "daily_weather": {}
        }

        for date in trip_dates:
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
                "summary": analysis.summary,
                "chance_of_rain": analysis.chance_of_rain,
                "advice": analysis.advice,
                "general_packing_tips": analysis.general_packing_tips
            }

        return result

# Main entry
if __name__ == "__main__":
    planner = WeatherPlanner(city="Goa", start_date="2025-04-24", end_date="2025-04-28")
    weather_plan = planner.plan_trip()

    print(json.dumps(weather_plan, indent=4))