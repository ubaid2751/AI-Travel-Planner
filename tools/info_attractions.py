from typing import List
from pydantic import BaseModel, Field
from groq import Groq
import instructor
import wikipedia

# Schema for output
class Attraction(BaseModel):
    name: str = Field(alias="name_of_site")
    description: str = Field(alias="description_of_site")
    type: str = Field(alias="type_of_site")
    estimated_time_hours: str = Field(alias="estimated_hours_to_spend")
    tags: List[str] = Field(default_factory=list, alias="activity_tags")

class AttractionPlaceInfo:
    def __init__(self, place_name: str):
        self.place_name = place_name
        self.client = instructor.from_groq(
            Groq(),
            mode=instructor.Mode.JSON,
        )

    def build_prompt(self, summary: str) -> str:
        return f"""
            You are a travel guide assistant helping users learn about tourist attractions.

            The user is interested in **{self.place_name}**.

            Based on the information below, provide a JSON object with:
            - A brief **description**
            - The **type** of place (e.g. historical, natural, theme park, museum)
            - **Estimated time to spend** there for a good memory
            - Optional: **Tags** like "indoor/outdoor", "family-friendly", etc.

            INFORMATION:
            {summary}

            Only respond with the JSON object as per the required structure.
        """

    def fetch_attraction_info(self) -> Attraction:
        try:
            summary = wikipedia.summary(self.place_name, sentences=5)
        except Exception as e:
            summary = f"Could not retrieve summary for {self.place_name}."

        prompt = self.build_prompt(summary)

        dest_info = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            response_model=Attraction,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that strictly follows the user’s instructions and format."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )

        return dest_info

    def print_info(self):
        dest_info = self.fetch_attraction_info()
        print(dest_info.model_dump_json(indent=4, by_alias=True))

if __name__ == "__main__":
    attraction_info = AttractionPlaceInfo("Jantar Mantar")
    attraction_info.print_info()
