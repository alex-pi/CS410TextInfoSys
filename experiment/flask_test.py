from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def start():
    process = subprocess.Popen('scrapy crawl links -a start_url=https://www.illinois.edu',
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE )
    return 'Hola'


if __name__ == '__main__':
    app.run(debug=True)