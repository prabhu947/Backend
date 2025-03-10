from groq import Groq
import base64
from dotenv import load_dotenv
# Function to encode the image
import os
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "./dbschema.png"

# Getting the base64 string
base64_image = encode_image(image_path)

# Initialize the Groq client with the API key
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "write code to create the schema to table using python in postgresql database for all the table with proper relation as given"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
    model="llama-3.2-11b-vision-preview",
)

print(chat_completion.choices[0].message.content)
