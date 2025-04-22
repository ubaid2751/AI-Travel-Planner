from utils.prompt_parser import PromptParser
from utils.weather_checker import WeatherPlanner
from utils.destination_info import DestinationInfo
from utils.attraction_info import AttractionPlaceInfo
import json

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
        self.city_info = dest_info.fetch_destination_info()
        
        # print("[Step 4] Getting detailed info for attractions...")
        # get_attraction_info = AttractionPlaceInfo()
        # for place in self.city_info["nearby_attractions"]:
        #     summary = get_attraction_info(place, self.parsed.city)
        #     self.attraction_details.append({
        #         "name": place,
        #         "summary": summary
        #     })

        return self.build_final_itinerary()

    def build_final_itinerary(self):
        return {
            "city": self.parsed.city,
            "trip_dates": self.weather["trip_dates"],
            "city_overview": self.city_info["city_description"],
            "weather_plan": self.weather["daily_weather"],
            "top_attractions": self.attraction_details
        }

# Example Usage
if __name__ == "__main__":
    prompt = "Plan a 3-day trip to Mumbai starting April 27"
    agent = TravelAgent(prompt)
    result = agent.run()

    print("\n🧳 Final Trip Plan:\n")
    print(json.dumps(result, indent=4))

    # Optionally save to file
    with open("final_itinerary.json", "w") as f:
        json.dump(result, f, indent=4)