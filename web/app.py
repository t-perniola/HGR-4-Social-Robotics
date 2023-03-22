# import dependancies
import utilities as utils
from flask import Flask, render_template, Response
#from pepper_app import pepper_app
from flask_ngrok import run_with_ngrok


# Declarations
app = Flask(__name__)
#app.register_blueprint(pepper_app, url_prefix = "/pepper")

# bool
open = False

# Use NGROK
#run_with_ngrok(app)


# Define ROUTES
@app.route("/")
def home():    
    return render_template("home.html")

@app.route('/webcam')
def webcam():
    return render_template("webcam.html")

@app.route('/pepper/')
def pepper():
    return render_template("peppercam.html")

@app.route('/video_feed')
def video_feed():
    open = True
    if open == True:
        return Response(utils.get_prediction_hagrid(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/video_feed_pepper')
def video_feed_pepper():
    open = True
    if open == True:
        return Response(utils.get_prediction(), mimetype='multipart/x-mixed-replace; boundary=frame')    


if __name__ == "__main__":
    app.run()

# TODO 1): capire come stoppare webcam
# TODO 2): estrarre il risultato del processo di riconoscimento (la prediction finale del gesto eseguito) 
