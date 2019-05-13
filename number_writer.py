import png
from number_generator import NumberGenerator
from flask import Flask, send_file, render_template, request
import numpy as np
import os

generator = NumberGenerator()

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

os.makedirs("tmp/img/", exist_ok=True)

@app.route("/")
def index():
    return render_template("number_generator.html")
    
@app.route("/yay")
def yay():
    img = generator.generate_number([0,1,2,3,4,5,6,7,8,9])
    img = (255 - img * 255).astype(np.uint8)
    print(img)
    png.from_array(img, "L").save("tmp/img/image.png")
    return send_file('tmp/img/image.png', attachment_filename='python.jpg')
    
@app.route("/generate", methods=["GET"])
def generate():
    numbers = [int(x) for x in request.args["number"]]
    ree = generator.generate_number(numbers)
    print(ree.shape)
    img = (255 - ree * 255).astype(np.uint8)
    png.from_array(img, "L").save("tmp/img/image.png")
    return send_file("tmp/img/image.png", attachment_filename="number.png")
    
    
    
app.run("0.0.0.0", 8007)