import os
import base64

import requests
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
import matplotlib

matplotlib.use('agg')

#### Get the dataset
data = pd.read_csv("data/db.csv")
DATA_PATH = "./static/images"

#### Function: interpret graphs-------------------------------------------------
# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def call_interpret(image_path, OPENAI_API_KEY):

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    payload = {
    "model": "gpt-4-turbo",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            #"text": "What’s in this image?"
            "text": "Could you interprete this image?"
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    json_response = response.json()
    content = json_response['choices'][0]['message']['content']
    return content
#### End Function: interpret graphs---------------------------------------------

#### HTML (search) -------------------------------------------------------------
def search(llm, query, OPENAI_API_KEY):

    # SmartDataframe configuration
    df = SmartDataframe(
        data,
        {"enable_cache": False},
        config={
            "llm": llm,
            "save_charts": True,
            "open_charts": False,
            "save_charts_path": DATA_PATH
        }
    )

    # Get the answer and add it to the list of answers
    answer = df.chat(query)

    img_interpretation = None

    # Check if the response is an image or text. If it's a path, it's an image.
    if os.path.isfile(answer):
        img_interpretation = call_interpret(answer, OPENAI_API_KEY)

    return img_interpretation,answer
#### HTML (search) -------------------------------------------------------------
