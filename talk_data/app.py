import base64
import os
import matplotlib
import pandas as pd
import requests

from dotenv import load_dotenv
from flask import Flask, render_template, request


from pandasai.llm.openai import OpenAI
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI


# app = Flask(__name__, template_folder="../frontend/temp__html/templates")
app = Flask(__name__)
matplotlib.use("agg")

data = pd.read_csv("data/db.csv")
DATA_PATH = "./static/images"

load_dotenv()

openai_key = os.getenv("OPENAI_KEY")

app = Flask(__name__, static_folder="static")
llm = OpenAI(api_token=openai_key)


def search_question(llm, query, openai_key):

    # SmartDataframe configuration
    df = SmartDataframe(
        data,
        {"enable_cache": False},
        config={
            "llm": llm,
            "save_charts": True,
            "open_charts": False,
            "save_charts_path": DATA_PATH,
        },
    )

    # Get the answer and add it to the list of answers
    answer = df.chat(query)

    img_interpretation = None

    # Check if the response is an image or text. If it's a path, it's an image.
    if os.path.isfile(str(answer)):
        img_interpretation = call_interpret(answer, openai_key)

    return img_interpretation, answer


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def call_interpret(image_path, openai_key):

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_key}",
    }

    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        # "text": "What’s in this image?"
                        "text": "Could you interprete this image?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
    )

    json_response = response.json()
    if "choices" in json_response:
        content = json_response["choices"][0]["message"]["content"]
        return content
    return "Could not interpret this"


@app.route("/")
def index():
    return render_template(
        "index.html", query=None, texto=None, img=None, frases=None
    )


@app.route("/search", methods=["POST"])
def search():
    query = request.form["query"]  # Get the query from the user
    interpretation, answer = search_question(llm, query, openai_key)

    if os.path.isfile(str(answer)):
        return render_template(
            "index.html",
            query=query,
            texto=interpretation,
            img=os.path.basename(answer),
            frases=None,
        )
    else:
        return render_template(
            "index.html", query=query, texto=answer, img=None, frases=None
        )


# Search Ideas
@app.route("/search_ideas", methods=["POST"])
def search_ideas():
    frases = [
        "What are the Species in our dataset",
        "Pie chart of the Species",
        "Average of the Value",
    ]
    return render_template("index.html", texto=None, img=None, frases=frases)


# Download
@app.route("/download", methods=["POST"])
def download():
    pass


if __name__ == "__main__":
    app.run(debug=True, threaded=False)


# End Function: interpret graphs---------------------------------------------


# HTML (search) -------------------------------------------------------------


# HTML (search) -------------------------------------------------------------
