import numpy as np
import json
import _thread as thread
from flask import Flask, render_template, request
import os

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
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
    #image = np.around(gan.x_train[0] * 1000)
    #return "[" + ",".join(str(img.tolist()) for img in image) + "]"
    return str(gan.training_iterations) + "\n" + str(image.tolist()[0])

@app.route("/model/predict", methods=["POST"])
def predict_image():
    image = np.array(request.get_json()["grid"])
    return str(json.loads("[%f]" % gan.predict_image(image)[0].item()))

@app.route("/model/train", methods=["GET"])
def train():
    gan.train(5, 64)
    return ""
    
@app.route("/getFirstImage")
def get_first_image():
    return str(gan.x_train[0].tolist())
    
@app.route("/static/images/")
def images():
    return "<br>".join(sorted(map(lambda f : "<a href='%s'>%s</a>" % (f, f), os.listdir("static/images/"))))

def set_GAN(GAN):
    global gan
    gan = GAN

def run_server():
    app.run("0.0.0.0", 8007)

def start_server():
    thread.start_new_thread(run_server, ())
    