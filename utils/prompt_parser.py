from datetime import datetime
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from groq import Groq
from instructor import from_groq, Mode

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama-3.3-70b-versatile"

class PromptParseResponse(BaseModel):
    city: str
    start_date: str
    end_date: str
    number_of_days: int

class PromptParser:
    def __init__(self):
        self.client = from_groq(
            Groq(api_key=GROQ_API_KEY),
            mode=Mode.JSON
        )

    def parse(self, prompt: str) -> PromptParseResponse:
        today = datetime.now().date().strftime("%Y-%m-%d")

        instructions = f"""
            You are a travel planning assistant.

            Extract the following details from the user prompt:
            - city: the name of the destination city
            - start_date: start date of the trip in yyyy-mm-dd format (assume {today} + 7 days if not specified)
            - end_date: end date in yyyy-mm-dd format (assume start_date + 2 days if not specified)
            - number_of_days: total duration of the trip

            Ensure the response is strictly valid JSON and adheres to this schema.
        """

        response = self.client.chat.completions.create(
            model=MODEL,
            response_model=PromptParseResponse,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that only returns structured JSON."},
                {"role": "user", "content": instructions + "\n\nUser Prompt:\n" + prompt},
            ],
            temperature=0.3
        )

        return response

# Example usage
if __name__ == "__main__":
    parser = PromptParser()
    result = parser.parse("Plan a 4-day trip to Goa from April 27")
    print(result.model_dump_json(indent=4))