import numpy as np
import json
import _thread as thread
from flask import Flask, render_template, request

app = Flask(__name__)
gan = None

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")
    
@app.route("/model/generate", methods=["GET"])
def generate_image():
    return str(gan.generate_image().tolist())

@app.route("/model/predict", methods=["POST"])
def predict_image():
    print(request.data)
    image = np.array(json.loads(request.data))
    return str(gan.predict_image(image))

@app.route("/model/train", methods=["GET"])
def train():
    gan.train()
    return ""

def set_GAN(GAN):
    global gan
    gan = GAN

def run_server():
    app.run("0.0.0.0", 8007)

def start_server():
    thread.start_new_thread(run_server, ())
    