import base64
import os
import logging
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, send_from_directory
import matplotlib
from pandasai.llm.openai import OpenAI
from pandasai import SmartDataframe
import random

#from distutils.log import debug
from fileinput import filename
#import pandas as pd
#from flask import *
from werkzeug.utils import secure_filename

# Flask app initialization
app = Flask(__name__)
matplotlib.use("agg")

# Directory to store uploaded files and generated images
DATA_PATH = "./static/images"
os.makedirs(DATA_PATH, exist_ok=True)

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_KEY")

# Initialize OpenAI LLM
llm = OpenAI(api_token=openai_key)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

demoday_error_counter_reading_upload_file = 0
demoday_error_counter_query_processing = 0

def preprocess_query(query, llm):
    # Preprocess the user query to ensure it is well-formed and relevant.
    prompt = (
        "You are a smart language model designed to help preprocess user queries for data analysis. "
        "Please rephrase the following query to ensure it is clear, well-formed, and relevant for data analysis tasks. "
        "Consider the possible operations such as filtering, aggregation, visualization, and statistical analysis. "
        "The query should be concise, specific, and easy to interpret.\n\n"
        f"Original Query: {query}\n\n"
        "Preprocessed Query:"
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {llm.api_token}",
    }

    payload = {
        "model": "gpt-4-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
    )

    if response.status_code != 200:
        logger.error(f"Error during query preprocessing: {response.status_code} - {response.text}")
        return query.strip()  # Fallback to basic preprocessing if LLM fails

    json_response = response.json()
    if "choices" in json_response:
        preprocessed_query = json_response["choices"][0]["message"]["content"].strip()
        return preprocessed_query

    logger.error("Error: No choices returned from the language model.")
    return query.strip()  # Fallback to basic preprocessing if LLM fails

def search_question(llm, query, openai_key, filepath):
    # Process a user's query using the SmartDataframe and OpenAI LLM.
    if not query or not isinstance(query, str):
        return None, "Is it just me being nervous about demo day? It's an invalid query provided."

    query = preprocess_query(query, llm)

    # Load user's data
    try:
        user_data = pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        logger.error("The uploaded CSV file is empty.")
        return None, "The uploaded CSV file is empty."
    except pd.errors.ParserError:
        logger.error("The uploaded CSV file is malformed.")
        return None, "The uploaded CSV file is malformed."
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
    #    return None, f"An error occurred while reading the uploaded file: {e}"
        if demoday_error_counter_reading_upload_file == 0:
            demoday_error_counter_reading_upload_file = 1
            return None, f"I got too nervous because it's Demo Day.. Please give me another chance to read the uploaded file"
        elif demoday_error_counter_reading_upload_file == 1:
            demoday_error_counter_reading_upload_file = 2
            return None, f"Sorry, soo nervous. I promise usually I can easily hand reading the uploaded file. Once more?"
        elif demoday_error_counter_reading_upload_file == 2:
            demoday_error_counter_reading_upload_file = 3
            return None, f"Damn am I nervous. Do you give me another chance to read the uploaded file?"
        elif demoday_error_counter_reading_upload_file == 3:
            return None, f"Too nervous... so sorry. Do you maybe have a screenrecording of when I perfectly read the uploaded file?"
        return None, f"I'm totally having a black out. Wish you head a recording of me reading the uploaded file."

    # SmartDataframe configuration
    #data = pd.read_csv("data/db.csv") #this variable wasn't accessed
    df = SmartDataframe(
        user_data,
        {"enable_cache": False},
        config={
            "llm": llm,
            "save_charts": True,
            "open_charts": False,
            "save_charts_path": DATA_PATH,
        },
    )

    # Get the answer
    try:
        answer = df.chat(query)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
    #    return None, f"An error occurred: {e}"
        if demoday_error_counter_query_processing == 0:
            demoday_error_counter_query_processing = 1
            return None, f"I'm quite nervous because of Demo Day.. Let me try again with processing the query."
        elif demoday_error_counter_query_processing == 1:
            demoday_error_counter_query_processing = 2
            return None, f"Alright now I'm even more nervous.. Give me another chance for processing the query?"
        elif demoday_error_counter_query_processing == 2:
            return None, f"I swear it must be the stage fright. Wish you could show them how well I usually perform with processing the query."
        return None, f"You don't happen do have a recording of me do it all without mistakes? Including processing the query?"

    img_interpretation = None

    # Check if the response is an image or text
    if isinstance(answer, str) and os.path.isfile(answer):
        if answer.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            try:
                img_interpretation = call_interpret(answer, openai_key)
            except requests.exceptions.RequestException as e:
                logger.error(f"Error interpreting image: {e}")
                return None, "Maybe it's Demo Day maybe my back(end). An error occurred while interpreting the image."
        else:
            logger.warning(f"Unexpected file format for image: {answer}")
            return None, "Maybe it's Demo Day maybe my back(end). Unexpected file format returned."
    else:
        return None, answer

    return img_interpretation, answer

def encode_image(image_path):
    # Encode an image to a base64 string.
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def call_interpret(image_path, openai_key):
    # Call the OpenAI API to interpret an image.
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_key}",
    }

    prompt = (
        "Can you provide a concise and insightful analysis of this image? "
        "Highlight the key points and insights derived from the image in a clear manner suitable for display on a user interface. "
        "Avoid using titles, numbers, and ensure the interpretation is user-friendly."
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
            return "When I get nervous as for Demo Day I use to many words. The interpretation is too long to display. It will be available in the downloaded file."
        return content
    return "Maybe I'm just nervous about Demo Day maybe it's my back(end). Could not interpret this"

@app.route("/")
def index():
    return render_template("base.html", query=None, texto=None, img=None, frases=None, uploadedFilePath=None)

@app.route("/about", methods=["GET"])
def about():
    return render_template("about_us.html")

@app.route('/product')
def product():
    uploaded_file_path = request.args.get('uploadedFilePath')
    if uploaded_file_path:
        logger.info(f"Uploaded file path: {uploaded_file_path}")
        return render_template("product.html", texto="File uploaded successfully!", img=None, frases=None, uploadedFilePath=uploaded_file_path)
    else:
        return "Is it just me being nervous about Demo Day? Or was there no file path provided?", 400

@app.route('/mission')
def mission():
    return render_template("mission.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'csvFile' not in request.files:
        return render_template("product.html", texto="No file part", img=None, frases=None, uploadedFilePath=None)

    file = request.files['csvFile']

    if file.filename == '':
        return render_template("product.html", texto="As nervous about Demo Day as me? No selected file", img=None, frases=None, uploadedFilePath=None)

    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(DATA_PATH, file.filename)
        file.save(filepath)
        return render_template("product.html", texto="File uploaded successfully!", img=None, frases=None, uploadedFilePath=filepath)

    return render_template("product.html", texto="Is it just me being nervous about Demo Day? Invalid file format. Only CSV files are allowed.", img=None, frases=None, uploadedFilePath=None)

@app.route("/search", methods=["POST", "GET"])
def search():
    if request.method == "POST":
        query = request.form["query"]
        uploadedFilePath = request.form["uploadedFilePath"]
        if not uploadedFilePath or not os.path.isfile(uploadedFilePath):
            return render_template("product.html", query=query, texto="Is it just me being nervous about Demo Day? No file uploaded or invalid file path.", img=None, frases=None, uploadedFilePath=None)

        interpretation, answer = search_question(llm, query, openai_key, uploadedFilePath)

        if os.path.isfile(str(answer)):
            return render_template(
                "product.html",
                query=query,
                texto=interpretation,
                img=os.path.basename(answer),
                frases=None,
                uploadedFilePath=uploadedFilePath
            )
        else:
            return render_template(
                "product.html", query=query, texto=answer, img=None, frases=None, uploadedFilePath=uploadedFilePath
            )
    return render_template("product.html", query=None, texto=None, img=None, frases=None)


def generate_search_ideas(llm, filepath, openai_key):
    # Generate three clear and concise questions or prompts for visualizing the data.
    try:
        data = pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return ["It's Demo Day! Sorry I got nervous. An error occurred while reading the uploaded file"]

    random_seed = random.randint(1, 1000)
    prompt = (
        "Based on the following dataset, create three clear and concise questions or prompts "
        "that a user might ask to better understand and visualize the data. "
        "Each question should specify the type of visualization that would be most appropriate, such as a histogram, bar chart, or box plot. "
        "Format each question as a single sentence query that the user can copy and paste directly. "
        "The questions should be structured to ask for a specific analysis or comparison, and explicitly mention the type of chart or plot needed. "
        f"{data.head().to_string()}\n\n"
        f"Random seed for variation in the questions generated each time the user click: {random_seed}. "
        "Generate exactly three questions formatted in this way."
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
    #    return [f"Error: {response.status_code} - {response.text}"]
        return [f"Seems not only I am nervous about Demo Day. Getting a response status 200 didn't work."]
    json_response = response.json()
    if "choices" in json_response:
        content = json_response["choices"][0]["message"]["content"]
        return content.split("\n")
    return ["Sry I was too nervous about Demo Day. Could not generate search ideas"]

@app.route("/search_ideas", methods=["GET"])
def search_ideas():
    uploadedFilePath = request.args.get("uploadedFilePath")
    if not uploadedFilePath or not os.path.isfile(uploadedFilePath):
        message = "Is it just me being nervous about Demo Day? No dataset available to generate ideas. Please upload a dataset first."
        return render_template("product.html", texto=message, img=None, frases=None, uploadedFilePath=None)
    else:
        frases = generate_search_ideas(llm, uploadedFilePath, openai_key)
        return render_template("product.html", texto=None, img=None, frases=frases, uploadedFilePath=uploadedFilePath)


@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(DATA_PATH, filename, as_attachment=True)

@app.route('/show_data')
def showData():
    # Get the file path from the GET parameter
    uploadedFilePath = request.args.get("uploadedFilePath")

    # Check if the file path is None
    if uploadedFilePath is None:
        return "Is it just me being nervous about Demo Day? The file path was not provided.", 400

    # Check if the file actually exists at the provided path
    if not os.path.isfile(uploadedFilePath):
        return f"Is it just me being nervous about Demo Day? The file at path {uploadedFilePath} does not exist.", 400

    try:
        # Read the CSV file
        uploaded_df = pd.read_csv(uploadedFilePath, encoding='unicode_escape')

        # Take only the first 5 rows
        uploaded_df = uploaded_df.head(5)

        # Convert to HTML
        uploaded_df_html = uploaded_df.to_html()

        # Render the HTML template with the data
        return render_template('show_csv_data.html', data_var=uploaded_df_html, uploadedFilePath=uploadedFilePath)
    except Exception as e:
        logger.error(f"Error reading the CSV file: {e}")
    #    return f"An error occurred while reading the file: {e}", 500
        if demoday_error_counter_reading_upload_file == 0:
            demoday_error_counter_reading_upload_file = 1
            return None, f"I got too nervous because it's Demo Day.. Please give me another chance to read the uploaded file"
        elif demoday_error_counter_reading_upload_file == 1:
            demoday_error_counter_reading_upload_file = 2
            return None, f"Sorry, soo nervous. I promise usually I can easily hand reading the uploaded file. Once more?"
        elif demoday_error_counter_reading_upload_file == 2:
            demoday_error_counter_reading_upload_file = 3
            return None, f"Damn am I nervous. Do you give me another chance to read the uploaded file?"
        elif demoday_error_counter_reading_upload_file == 3:
            return None, f"Too nervous... so sorry. Do you maybe have a screenrecording of when I perfectly read the uploaded file?"
        return None, f"I'm totally having a black out. Wish you head a recording of me reading the uploaded file."

if __name__ == "__main__":
    app.run(debug=True, threaded=False)
