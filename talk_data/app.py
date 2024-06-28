import base64
import os
import matplotlib
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, send_file

from pandasai.llm.openai import OpenAI
from pandasai import SmartDataframe

app = Flask(__name__)
matplotlib.use("agg")

data = pd.read_csv("data/db.csv")
DATA_PATH = "./static/images"

load_dotenv()

openai_key = os.getenv("OPENAI_KEY")

# Ensure the API key is loaded
if not openai_key:
    raise ValueError("OpenAI API key not found in environment variables.")

app = Flask(__name__, static_folder="static")
llm = OpenAI(api_token=openai_key)

def search_question(llm, query, openai_key):
    df = SmartDataframe(data, config={"llm": llm, "save_charts": True, "open_charts": False, "save_charts_path": DATA_PATH})
    answer = df.chat(query)
    img_interpretation = None

    if os.path.isfile(str(answer)):
        img_interpretation = call_interpret(answer, openai_key)
    return img_interpretation, answer

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def call_interpret(image_path, openai_key):
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_key}",
    }

    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {"role": "user", "content": "Could you interpret this image?"},
            {"role": "user", "content": f"data:image/jpeg;base64,{base64_image}"}
        ],
        "max_tokens": 300,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
    )

    if response.status_code != 200:
        return f"Error: {response.status_code} - {response.text}"

    json_response = response.json()
    if "choices" in json_response:
        content = json_response["choices"][0]["message"]["content"]
        return content
    return "Could not interpret this"

def generate_search_ideas(llm, data, openai_key):
    prompt = (
        "Given the following dataset, generate some interesting questions or prompts "
        "that a user might want to ask to understand and visualize the data better:\n\n"
        f"{data.head().to_string()}\n\n"
        "Please provide at least five questions or prompts."
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

@app.route("/")
def index():
    return render_template("index.html", query=None, texto=None, img=None, frases=None)

@app.route("/search", methods=["POST"])
def search():
    query = request.form["query"]
    interpretation, answer = search_question(llm, query, openai_key)

    if os.path.isfile(str(answer)):
        return render_template(
            "index.html", query=query, texto=interpretation, img=os.path.basename(answer), frases=None, feedback_step=True
        )
    else:
        return render_template(
            "index.html", query=query, texto=answer, img=None, frases=None, feedback_step=True
        )

@app.route("/feedback", methods=["POST"])
def feedback():
    feedback_text = request.form["feedback"]
    original_query = request.form["original_query"]
    enhanced_query = f"{original_query} Based on the user's feedback: {feedback_text}"
    interpretation, answer = search_question(llm, enhanced_query, openai_key)

    if os.path.isfile(str(answer)):
        return render_template(
            "index.html", query=enhanced_query, texto=interpretation, img=os.path.basename(answer), frases=None
        )
    else:
        return render_template(
            "index.html", query=enhanced_query, texto=answer, img=None, frases=None
        )

@app.route("/search_ideas", methods=["GET"])
def search_ideas():
    frases = generate_search_ideas(llm, data, openai_key)
    return render_template("index.html", texto=None, img=None, frases=frases)

@app.route("/download", methods=["GET"])
def download():
    return send_file('queries_answers.csv', mimetype='text/csv', as_attachment=True, download_name='queries_answers.csv')

if __name__ == "__main__":
    app.run(debug=True)
