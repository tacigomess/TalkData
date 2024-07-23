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
from werkzeug.utils import secure_filename
from functools import lru_cache

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

# Cache for storing processed results
cache = {}

def preprocess_query(query, llm):
    """
    Preprocess the user query to ensure it is well-formed and relevant.

    Args:
        query (str): The user's query.
        llm (OpenAI): The OpenAI language model instance.

    Returns:
        str: The preprocessed query.
    """
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

def load_user_data(filepath):
    """
    Loads the user's data from the specified CSV file.

    Args:
        filepath (str): The path to the user's uploaded CSV file.

    Returns:
        DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        return pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        logger.error("The uploaded CSV file is empty.")
    except pd.errors.ParserError:
        logger.error("The uploaded CSV file is malformed.")
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
    return None

def configure_smart_dataframe(user_data, llm):
    """
    Configures the SmartDataframe with the user's data and LLM.

    Args:
        user_data (DataFrame): The user's data as a pandas DataFrame.
        llm (OpenAI): The OpenAI language model instance.

    Returns:
        SmartDataframe: The configured SmartDataframe instance.
    """
    return SmartDataframe(
        user_data,
        {"enable_cache": True},
        config={
            "llm": llm,
            "save_charts": True,
            "open_charts": False,
            "save_charts_path": DATA_PATH,
        },
    )

def get_answer_from_df(df, query):
    """
    Retrieves the answer from the SmartDataframe based on the query.

    Args:
        df (SmartDataframe): The SmartDataframe instance.
        query (str): The user's query.

    Returns:
        str: The answer to the query.
    """
    try:
        return df.chat(query)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return None

def interpret_image_if_needed(answer, openai_key):
    """
    Interprets the image if the answer is an image path.

    Args:
        answer (str): The answer from the SmartDataframe.
        openai_key (str): The API key for OpenAI.

    Returns:
        str: The image interpretation if applicable.
    """
    if isinstance(answer, str) and os.path.isfile(answer):
        if answer.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            try:
                return call_interpret(answer, openai_key)
            except requests.exceptions.RequestException as e:
                logger.error(f"Error interpreting image: {e}")
                return None
        else:
            logger.warning(f"Unexpected file format for image: {answer}")
    return None

def cache_key(query, filepath):
    """
    Generates a cache key based on the query and filepath.

    Args:
        query (str): The user's query.
        filepath (str): The path to the user's uploaded CSV file.

    Returns:
        str: The generated cache key.
    """
    return f"{query}:{filepath}"

def search_question(llm, query, openai_key, filepath):
    """
    Processes a user's query using the SmartDataframe and OpenAI LLM.

    Args:
        llm (OpenAI): The OpenAI language model instance.
        query (str): The user's query.
        openai_key (str): The API key for OpenAI.
        filepath (str): The path to the user's uploaded CSV file.

    Returns:
        tuple: A tuple containing image interpretation and the answer.
    """
    if not query or not isinstance(query, str):
        return None, "Invalid query provided."

    # Check cache first
    key = cache_key(query, filepath)
    if key in cache:
        logger.info("Cache hit for key: %s", key)
        return cache[key]

    # Preprocess the query
    query = preprocess_query(query, llm)

    # Load user's data
    user_data = load_user_data(filepath)
    if user_data is None:
        return None, "An error occurred while reading the uploaded file."

    # SmartDataframe configuration
    df = configure_smart_dataframe(user_data, llm)

    # Get the answer
    answer = get_answer_from_df(df, query)
    if answer is None:
        return None, "An error occurred while processing the query."

    # Interpret the response if it is an image
    img_interpretation = interpret_image_if_needed(answer, openai_key)

    # Cache the result
    cache[key] = (img_interpretation, answer)

    return img_interpretation, answer

def encode_image(image_path):
    """
    Encode an image to a base64 string.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def call_interpret(image_path, openai_key):
    """
    Call the OpenAI API to interpret an image.

    Args:
        image_path (str): The path to the image file.
        openai_key (str): The API key for OpenAI.

    Returns:
        str: The interpretation of the image.
    """
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_key}",
    }

    prompt = (
        "Can you provide a concise and insightful analysis of this image? "
        "Highlight the key points and insights derived from the image in a clear manner suitable for display on a user interface. "
        "Avoid using titles, numbers, and ensure the interpretation is user-friendly."
        "Don't write more than 150 words."
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
        return "No file path provided", 400

@app.route('/mission')
def mission():
    return render_template("mission.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'csvFile' not in request.files:
        return render_template("product.html", texto="No file part", img=None, frases=None, uploadedFilePath=None)

    file = request.files['csvFile']

    if file.filename == '':
        return render_template("product.html", texto="No selected file", img=None, frases=None, uploadedFilePath=None)

    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(DATA_PATH, file.filename)
        file.save(filepath)
        return render_template("product.html", texto="File uploaded successfully!", img=None, frases=None, uploadedFilePath=filepath)

    return render_template("product.html", texto="Invalid file format. Only CSV files are allowed.", img=None, frases=None, uploadedFilePath=None)

@app.route("/search", methods=["POST", "GET"])
def search():
    if request.method == "POST":
        query = request.form["query"]
        uploadedFilePath = request.form["uploadedFilePath"]
        if not uploadedFilePath or not os.path.isfile(uploadedFilePath):
            return render_template("product.html", query=query, texto="No file uploaded or invalid file path.", img=None, frases=None, uploadedFilePath=None)

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
    """
    Generate three clear and concise questions or prompts for visualizing the data.

    Args:
        llm (OpenAI): The OpenAI language model instance.
        filepath (str): The path to the user's uploaded CSV file.
        openai_key (str): The API key for OpenAI.

    Returns:
        list: A list of generated questions or prompts.
    """
    try:
        data = pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return ["An error occurred while reading the uploaded file"]

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
        "One of the questions should be: Generate a line graph depicting the annual trend of 'Previous Policies In Force Quantity' using 'Profile Date Year' as the time axis."
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

@app.route("/search_ideas", methods=["GET"])
def search_ideas():
    uploadedFilePath = request.args.get("uploadedFilePath")
    if not uploadedFilePath or not os.path.isfile(uploadedFilePath):
        message = "No dataset available to generate ideas. Please upload a dataset first."
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
        return "The file path was not provided.", 400

    # Check if the file actually exists at the provided path
    if not os.path.isfile(uploadedFilePath):
        return f"The file at path {uploadedFilePath} does not exist.", 400

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
        return f"An error occurred while reading the file: {e}", 500

if __name__ == "__main__":
    app.run(debug=True, threaded=False)
