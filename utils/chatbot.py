from google import genai
import config

client = genai.Client(
    api_key=config.GOOGLE_API_KEY
)

client = genai.Client(api_key=GOOGLE_API_KEY)


def ask_gemini(question: str) -> str:
    """
    Sends a question to Gemini and returns the answer.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=question
    )

    return response.text