import _thread as thread
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")
    
def run_server():
    app.run("0.0.0.0", 8007)

def start_server():
    thread.start_new_thread(run_server, ())
    