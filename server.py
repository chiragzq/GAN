import numpy as np
import json
import _thread as thread
from flask import Flask, render_template, request

app = Flask(__name__)
gan = None
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")
    
@app.route("/model/generate", methods=["GET"])
def generate_image():
    image = np.around(gan.generate_image() * 1000)
    return str(image.tolist()[0])

@app.route("/model/predict", methods=["POST"])
def predict_image():
    image = np.array(request.get_json()["grid"])
    return str(json.loads("[%f]" % gan.predict_image(image)[0].item()))

@app.route("/model/train", methods=["GET"])
def train():
    gan.train(20, 64)
    return ""

def set_GAN(GAN):
    global gan
    gan = GAN

def run_server():
    app.run("0.0.0.0", 8007)

def start_server():
    thread.start_new_thread(run_server, ())
    