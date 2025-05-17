import os
import json
from groq import Groq
from dotenv import load_dotenv

from utils.prompt_parser import PromptParser
from utils.weather_checker import WeatherPlanner
from utils.destination_info import DestinationInfo
from utils.attraction_info import AttractionPlaceInfo
from utils.itinerary_composer import ItineraryComposer

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class TravelAgent:
    def __init__(self, user_prompt):
        self.user_prompt = user_prompt
        self.parsed = None
        self.weather = None
        self.city_info = None
        self.attraction_details = []

    def run(self):
        print("[Step 1] Parsing user prompt...")
        self.parsed = PromptParser().parse(self.user_prompt)

        print("[Step 2] Fetching weather forecast...")
        self.weather = WeatherPlanner(
            city=self.parsed.city,
            start_date=self.parsed.start_date,
            end_date=self.parsed.end_date
        ).plan_trip()

        print("[Step 3] Fetching city description and attractions...")
        dest_info = DestinationInfo(self.parsed.city)
        raw_city_info = dest_info.fetch_destination_info()
        self.city_info = raw_city_info.model_dump()
        
        print("[Step 4] Getting detailed info for attractions...")
        for place in self.city_info["nearby_attractions"]:
            print(f"  - Processing: {place}")
            try:
                attraction_info = AttractionPlaceInfo(place).fetch_attraction_info()
                self.attraction_details.append(attraction_info.model_dump())
            except Exception as e:
                print(f"⚠️ Skipped {place} due to error.")
                continue

        return self.build_final_itinerary()

    def build_final_itinerary(self):
        return {
            "city": self.parsed.city,
            "trip_dates": self.weather["trip_dates"],
            "city_overview": self.city_info["city_description"],
            "weather_plan": self.weather["daily_weather"],
            "top_attractions": self.attraction_details
        }

    def get_itinerary(self):
        itinerary = self.run()
        composer = ItineraryComposer(itinerary)
        self.itinerary = composer.compose_itinerary()
        output = {
            "city": self.parsed.city,
            "itinerary": self.itinerary
        }
        return output

class Manager:
    def __init__(self, user_prompt):
        self.user_prompt = user_prompt
        self.agent = TravelAgent(self.user_prompt)
        self.itinerary = self.agent.get_itinerary()

    def run(self):
        prompt = f"""
        You are a travel planner assistant. I will give you the output of my itinerary in structured JSON. Based on this, generate a detailed and engaging travel plan for the user, day-by-day.

        Include:
        - A short intro about the city.
        - Day-wise plans (mention date, places to visit, weather summary, estimated time at each place).
        - Add packing tips and weather advice if needed.
        - Add any extra helpful travel tips or food recommendations.

        ONLY use the information given in the JSON.

        Here is the JSON:
        """
        self.client = Groq(api_key=GROQ_API_KEY)
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(self.itinerary)}
            ],
            stream=True
        )
        output = ""
        for chunk in response:
            output += chunk.choices[0].delta.content or ""
            print(chunk.choices[0].delta.content, end="", flush=True)

        return output
# Example Usage
if __name__ == "__main__":
    prompt = "Plan a 3-day trip to Nainital."
    # agent = TravelAgent(prompt)
    # result = agent.get_itinerary()

    # print("\n🧳 Final Trip Plan:\n")
    # print(json.dumps(result, indent=4))

    # # Optionally save to file
    # with open("final_itinerary.json", "w") as f:
    #     json.dump(result, f, indent=4)
    manager = Manager(prompt)
    output = manager.run()

    # Write a markdown file
    with open("output.md", "w") as f:
        f.write(output)