import base64
import os
import matplotlib
import pandas as pd
import requests
import random
from dotenv import load_dotenv
from flask import Flask, render_template, request

#from pandasai.llm.openai import OpenAI
#from pandasai import SmartDataframe

from distutils.log import debug
from fileinput import filename
import pandas as pd
from flask import *
from werkzeug.utils import secure_filename


# Flask app initialization
app = Flask(__name__)
matplotlib.use("agg")

# Load data
DATA_PATH = "./static/images"

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_KEY")

# Initialize OpenAI LLM
#llm = OpenAI(api_token=openai_key)


# Configuration of the function upload .csv file
UPLOAD_FOLDER = os.path.join('data')

# Define allowed files
ALLOWED_EXTENSIONS = {'csv'}

# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = 'This is your secret key to utilize session in Flask'



def search_question(llm, query, openai_key):
    # SmartDataframe configuration
    data = pd.read_csv("data/db.csv")
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

    # Get the answer
    answer = df.chat(query)
    img_interpretation = None

    # Check if the response is an image or text
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
        "Make sure to highlight key points and insights derived from the image. "
        "It's gonna be displayed on a user interface so display things accordingly and try to not use titles, numbers, and stuff."
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
        if word_count > 300:
            return "The interpretation is too long to display. It will be available in the downloaded file."
        return content
    return "Could not interpret this"


@app.route("/")
def index():
    return render_template("base.html", query=None, texto=None, img=None, frases=None)

@app.route('/product2')
def product2():
    return render_template("product2.html", texto=None, img=None, frases=None)

@app.route("/about", methods=["GET"])
def about():
    return render_template("about_us.html")


@app.route("/search", methods=["POST", "GET"])
def search():
    if request.method == "POST":
        query = request.form["query"]  # Get the query from the user

        # Check if the query is empty
        if not query.strip():
            return render_template("product2.html", query=None, texto="Type your query to start the search.", img=None, frases=None)

        interpretation, answer = search_question(llm, query, openai_key)

        if os.path.isfile(str(answer)):
            return render_template(
                "product2.html",
                query=query,
                texto=interpretation,
                img=os.path.basename(answer),
                frases=None,
            )
        else:
            return render_template(
                "product2.html", query=query, texto=answer, img=None, frases=None
            )
    return render_template("product.html", query=None, texto=None, img=None, frases=None)


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
    file_path = "data/db.csv"

    if os.path.exists(file_path):
        data = pd.read_csv("data/db.csv")
        frases = generate_search_ideas(llm, data, openai_key)
        return render_template("product2.html", texto=None, img=None, frases=frases)
    else:
        texto = "You need first a .csv file"
        return render_template("product.html", texto=None, img=None, frases=None)


# Upload .csv file
@app.route('/upload', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
      # upload file flask
        f = request.files.get('file')

        # static name
        fixed_filename = "db.csv"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], fixed_filename)

        # Extracting uploaded file name
        data_filename = secure_filename(f.filename)
        print("Name of the data_file: ")
        print(data_filename)

        #f.save(os.path.join(app.config['UPLOAD_FOLDER'],
        #                    data_filename))

        f.save(file_path)

        #session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'],
        #             data_filename)

        session['uploaded_data_file_path'] = file_path

        return render_template("product2.html", texto=None, img=None, frases=None)
    return render_template("product.html", texto=None, img=None, frases=None)

# TODO: how to return to the product.html without lose the data
@app.route('/show_data')
def showData():
    # Uploaded File Path
    data_file_path = session.get('uploaded_data_file_path', None)
    # read csv
    uploaded_df = pd.read_csv(data_file_path,
                              encoding='unicode_escape')

    uploaded_df = uploaded_df.head(5)

    # Converting to html Table
    uploaded_df_html = uploaded_df.to_html()
    return render_template('show_csv_data.html',
                           data_var=uploaded_df_html)


# Download
@app.route("/download", methods=["POST"])
def download():
    pass


if __name__ == "__main__":
    app.run(debug=True, threaded=False)



# End Function: interpret graphs---------------------------------------------


# HTML (search) -------------------------------------------------------------


# HTML (search) -------------------------------------------------------------
