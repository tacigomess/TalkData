
import base64
import os
import matplotlib
import pandas as pd
import requests
import random
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

    prompt = (
        "Could you interpret this image? Please provide a clear, understandable, "
        "and insightful interpretation that will be useful for our users. "
        "Make sure to highlight key points and insights derived from the image."
        "It's gonna be displayed on a user interface so display things accordingly and try to not use titles, numbers and stuff."
        "It should be easy to display"
    )

    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
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
        "max_tokens": 200,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
    )

    json_response = response.json()
    if "choices" in json_response:
        content = json_response["choices"][0]["message"]["content"]
        word_count = len(content.split())
        if word_count > 150:
            return "The interpretation is too long to display. It will be available in the downloaded file."
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
def generate_search_ideas(llm, data, openai_key):
    random_seed = random.randint(1, 1000)
    prompt = (
        "Given the following dataset, generate questions or prompts "
        "that a user might want to ask to understand and visualize the data better:\n\n"
        f"{data.head().to_string()}\n\n"
        "Here are three interesting questions or prompts for analyzing the given dataset. "
        "Please include questions suitable for generating charts or visualizations such as histograms, heatmaps, scatter plots, etc. "
        f"Random seed for variation in the questions generated each time the user click: {random_seed}. "
        "Format the output as a numbered list (1, 2, 3) and limit it to 3 for clarity."
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_key}",
    }

    payload = {
        "model": "gpt-4-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
    )

    if response.status_code != 200:
        return [f"Error: {response.status_code} - {response.text}"]

    json_response = response.json()
    if "choices" in json_response:
        content = json_response["choices"][0]["message"]["content"]
        return content.split("\n")
    return ["Could not generate search ideas"]


# Search Ideas
@app.route("/search_ideas", methods=["GET"])
def search_ideas():
    frases = generate_search_ideas(llm, data, openai_key)
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
