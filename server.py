from flask import Flask

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")
    
def start_server():
    app.run()
    