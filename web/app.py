from flask import Flask
from flask_ngrok import run_with_ngrok
  
app = Flask(__name__)
run_with_ngrok(app)
  
@app.route("/")
def home():
    return "ziocan"
  
if __name__ == "__main__":
  app.run()
