import os
from . import teste
from .search_ideas import search_ideas_options
from flask import Flask, render_template, request

from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv

app = Flask(__name__,
            static_folder='../frontend/temp_html/static',
            template_folder="../frontend/temp_html/templates")


load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

llm = OpenAI(api_token=OPENAI_API_KEY)

@app.route('/')
def index():
    return render_template('index.html', query= None, texto=None, img=None, frases=None)

@app.route('/search',methods=['POST'])
def search():
    query = request.form['query'] # Get the query from the user
    interpretation, answer = teste.search(llm, query, OPENAI_API_KEY)

    if os.path.isfile(answer):
        return render_template('index.html', query=query, texto=interpretation, img=os.path.basename(answer), frases=None)
    else:
        return render_template('index.html', query=query, texto=answer, img=None, frases=None)

## Search Ideas
@app.route('/search_ideas',methods=['POST'])
def search_ideas():
    frases = search_ideas_options()
    return render_template('index.html', texto=None, img=None, frases=frases)

## Download
@app.route('/download',methods=['POST'])
def download():
    pass

if __name__ == '__main__':
    app.run(debug=True, threaded=False)
