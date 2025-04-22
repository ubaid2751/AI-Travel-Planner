from textwrap import indent
import instructor
from pydantic import BaseModel, Field
from groq import Groq
import wikipedia
import json
from info_attractions import AttractionPlaceInfo

# Set Wikipedia language
wikipedia.set_lang("en")

# Define the structure of the response using Pydantic
class Destination(BaseModel):
    city_name: str = Field(alias="destination_city_name")
    city_description: str = Field(alias="destination_city_description")
    nearby_attractions: list[str] = Field(alias="nearby_attractions")

class DestinationInfo:
    def __init__(self, city_name: str):
        self.city_name = city_name
        self.client = instructor.from_groq(
            Groq(),
            mode=instructor.Mode.JSON,
        )

    def get_wikipedia_summary(self) -> str:
        return wikipedia.summary(self.city_name)

    def build_prompt(self, summary: str) -> str:
        return f"""
            You are a travel expert assistant helping users discover the best places to visit in different cities.

            The user is interested in **{self.city_name}**. Based on the following summary from Wikipedia, provide a response in the following structured format:

            - `destination_city_name`: Name of the destination i.e {self.city_name}.
            - `destination_city_description`: A short, engaging paragraph about why this place is worth visiting, based on the summary and general knowledge.
            - `nearby_attractions`: A list of 4 to 5 well-known nearby places (landmarks, beaches, historical sites, etc.) to explore around the destination.

            ### Wikipedia Summary:
            {summary}

            Only respond with the JSON object according to the structure above.
        """

    def fetch_destination_info(self) -> Destination:
        summary = self.get_wikipedia_summary()
        prompt = self.build_prompt(summary)

        dest_info = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            response_model=Destination,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that strictly follows the user’s instructions and format."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )

        return dest_info

    def print_info(self):
        dest_info = self.fetch_destination_info()
        print(dest_info.model_dump_json(indent=4))

if __name__ == "__main__":
    destination_handler = DestinationInfo("Jaipur")
    dest_info = destination_handler.fetch_destination_info()
    print(dest_info.model_dump_json(indent=4))