import cv2
import numpy as np
import mediapipe as mp
from flask import Blueprint, render_template, Response
from keras.models import load_model

pepper_app = Blueprint("pepper_app", __name__, static_folder="static", template_folder="templates")

mp_hands = mp.solutions.hands  # Hands model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

model = load_model(r"C:\Users\follo\OneDrive\Documenti\GitHub\HGR-4-Social-Robotics\models_action\camera.h5", compile=False)

labels = np.array(["preghiera", "saluto", "baci", "applauso"])
colors = [(245,117,16), (117,245,16), (16,117,245), (16,20,245)]

cap = cv2.VideoCapture(0)

@pepper_app.route("/home")
def home():
    return "<h1>home</h1>"

@pepper_app.route("/")
@pepper_app.route('/webcam')
def webcam():
    return render_template("peppercam.html")

@pepper_app.route('/video_feed_pepper')
def video_feed_pepper():
    return Response(get_prediction(), mimetype='multipart/x-mixed-replace; boundary=frame')

def get_prediction(): 
    sequence = []
    predictions = []
    gesto = ""
    threshold = 0.5
    
    # Set mediapipe model 
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5) as hands:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            if not ret:
                print("Ignoring empty pepper_app frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            
            # Make detections
            image, results = mediapipe_detection(frame, hands)                    
            #print("which hand?", results.multi_handedness)

            first_hand_keypoints = np.zeros(21*3)
            second_hand_keypoints = np.zeros(21*3)
        
            if results.multi_hand_landmarks:
                for num, hand_landmarks in enumerate(results.multi_hand_landmarks):        

                    mp_drawing.draw_landmarks(     
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                        )
                    
                    if num == 0:   
                        first_hand_keypoints = extract_keypoints_hands(results, hand_landmarks)
                        #print("\n1st hand kp:", first_hand_keypoints_test)
                    if num == 1:
                        second_hand_keypoints = extract_keypoints_hands(results, hand_landmarks)
                        #print("\n2nd hand kp:", second_hand_keypoints_test)

                keypoints = np.concatenate([first_hand_keypoints, second_hand_keypoints])    

            else: 
                keypoints = np.zeros(21*6)
                print("no detect")
                            
            # 2. Prediction logic
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]  
                predictions.append(np.argmax(res))
                        
            #3. Viz logic
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold:                     
                        gesto = labels[np.argmax(res)]
                
                # Viz probabilities
                image = prob_viz(res, labels, image, colors)
                
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(gesto), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')  # concat frame one by one and show result      

# funzioni ausiliari
def mediapipe_detection(image, model):
    # COLOR CONVERSION BGR 2 RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                  # Image is no longer writeable
    # Make prediction with the Holistic model
    results = model.process(image)
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results
def extract_keypoints_hands(results, hand_landmarks):
    h = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten() if results.multi_hand_landmarks else np.zeros(21*3)    
    return h
def prob_viz(res, labels_hagrid, input_frame, colors_hagrid):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60+num*40), (int(prob*100), 90+num*40), colors_hagrid[num], -1)
        cv2.putText(output_frame, labels_hagrid[num], (0, 85+num*40), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
