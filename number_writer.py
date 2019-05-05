import png
from number_generator import NumberGenerator
from flask import Flask, send_file, render_template, request
import numpy as np

generator = NumberGenerator()

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.route("/")
def index():
    return render_template("number_generator.html")
    
@app.route("/yay")
def yay():
    img = generator.generate_number([0,1,2,3,4,5,6,7,8,9])
    img = (255 - np.array(img) * 255).astype(np.uint8)
    print(img)
    png.from_array(img, "L").save("tmp/img/image.png")
    return send_file('tmp/img/image.png', attachment_filename='python.jpg')
    
@app.route("/generate", methods=["GET"])
def generate():
    numbers = [int(x) for x in request.args["number"]]
    img = (255 - np.array(generator.generate_number(numbers)) * 255).astype(np.uint8)
    png.from_array(img, "L").save("tmp/img/image.png")
    return send_file("tmp/img/image.png", attachment_filename="number.png")
    
    
    
app.run("0.0.0.0", 8007)