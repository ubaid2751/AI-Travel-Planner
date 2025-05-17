import json
from datetime import datetime
from typing import List, Dict, Any

class ItineraryComposer:
    def __init__(self, trip_plan: Dict[str, Any]):
        self.trip_plan = trip_plan
        self.city = trip_plan.get("city", "Unknown City")
        self.dates = trip_plan.get("trip_dates", [])
        self.weather_plan = trip_plan.get("weather_plan", {})
        self.attractions = trip_plan.get("top_attractions", [])
        self.itinerary = []

    def is_outdoor(self, attraction: Dict[str, Any]) -> bool:
        tags = [tag.lower() for tag in attraction.get("activity_tags", [])]
        # Consider attraction outdoor if tags contain 'outdoor'
        return "outdoor" in tags

    def compose_itinerary(self):
        # Simple heuristic:
        # - For rainy days (chance_of_rain > 30%), assign indoor attractions if possible
        # - Otherwise assign outdoor
        # - Distribute attractions evenly day-wise
        outdoor_attractions = [a for a in self.attractions if self.is_outdoor(a)]
        indoor_attractions = [a for a in self.attractions if not self.is_outdoor(a)]

        # Prepare itinerary list with empty attractions
        itinerary_map = {date: {"date": date, "weather": {}, "attractions": []} for date in self.dates}

        # Assign weather info
        for date in self.dates:
            weather = self.weather_plan.get(date, {})
            itinerary_map[date]["weather"] = {
                "summary": weather.get("summary", "No data"),
                "advice": weather.get("advice", ""),
                "packing_tips": weather.get("general_packing_tips", []),
            }

        # Split days by rain condition
        rainy_days = [d for d in self.dates if self.weather_plan.get(d, {}).get("chance_of_rain", "0%").rstrip('%').isdigit() 
                      and int(self.weather_plan[d]["chance_of_rain"].rstrip('%')) > 30]
        clear_days = [d for d in self.dates if d not in rainy_days]

        # Assign indoor attractions first to rainy days (one or two per day)
        idx = 0
        for day in rainy_days:
            if idx < len(indoor_attractions):
                itinerary_map[day]["attractions"].append(indoor_attractions[idx])
                idx += 1
            if idx < len(indoor_attractions):
                itinerary_map[day]["attractions"].append(indoor_attractions[idx])
                idx += 1

        # Assign outdoor attractions to clear days evenly
        idx = 0
        for day in clear_days:
            if idx < len(outdoor_attractions):
                itinerary_map[day]["attractions"].append(outdoor_attractions[idx])
                idx += 1
            if idx < len(outdoor_attractions):
                itinerary_map[day]["attractions"].append(outdoor_attractions[idx])
                idx += 1

        # If leftover attractions (either indoor or outdoor), assign to any days with less than 2 attractions
        leftovers = indoor_attractions[idx:] + outdoor_attractions[idx:]
        # First assign to days with zero attractions
        for day in self.dates:
            if len(itinerary_map[day]["attractions"]) == 0 and leftovers:
                itinerary_map[day]["attractions"].append(leftovers.pop(0))

        # Then fill days with fewer than 2 attractions
        for day in self.dates:
            while len(itinerary_map[day]["attractions"]) < 2 and leftovers:
                itinerary_map[day]["attractions"].append(leftovers.pop(0))


        # Convert map to sorted list by date
        self.itinerary = [itinerary_map[d] for d in sorted(self.dates)]
        return self.itinerary

    def save_to_json(self, filepath="Itinerary.json"):
        output = {
            "city": self.city,
            "itinerary": self.itinerary
        }
        with open(filepath, "w", encoding='utf-8') as f:
            json.dump(output, f, indent=4)
        print(f"Saved itinerary to {filepath}")

    def print_readable(self):
        print(f"Trip Itinerary for {self.city}:\n")
        for day in self.itinerary:
            date_str = day["date"]
            weather = day["weather"]
            attractions = day["attractions"]

            print(f"Date: {date_str}")
            print(f"  Weather: {weather.get('summary')}")
            print(f"  Advice: {weather.get('advice')}")
            if weather.get("packing_tips"):
                print(f"  Packing Tips: {', '.join(weather['packing_tips'])}")

            if attractions:
                print(f"  Attractions:")
                for a in attractions:
                    print(f"   - {a['name_of_site']} [{a['type_of_site']}, {a['estimated_hours_to_spend']}]")
                    desc = a.get("description_of_site", "")
                    if desc:
                        print(f"      {desc}")
            else:
                print("  No attractions planned for this day.")
            print("\n" + "-"*40 + "\n")

if __name__ == "__main__":
    # Load your trip plan JSON here (change filename if needed)
    with open("final_itinerary.json", "r") as file:
        trip_plan_data = json.load(file)

    composer = ItineraryComposer(trip_plan_data)
    composer.compose_itinerary()
    composer.save_to_json()
    composer.print_readable()