# import dependancies
import cv2
import mediapipe as mp
import numpy as np
import pickle
from flask import Flask, render_template, Response
from keras.models import load_model
#from flask_ngrok import run_with_ngrok


# Declarations
cap = cv2.VideoCapture(0)
app = Flask(__name__)
mp_hands = mp.solutions.hands  # Hands model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

# load models | TODO: rel paths
grid_search = pickle.load(open(r"C:\Users\follo\OneDrive\Documenti\GitHub\HGR-4-Social-Robotics\models_action\my_model_rndf.pickle", 'rb'))
model = load_model(r"C:\Users\follo\OneDrive\Documenti\GitHub\HGR-4-Social-Robotics\models_action\camera.h5", compile=False)

# define labels
labels_hagrid = np.array(["ok", "peace", "rock", "victory", "mano del gaucho", "saluto", "mute", "like", "dislike"])
labels = np.array(["preghiera", "saluto", "baci", "applauso"])

# define colors
colors = [(245,117,16), (117,245,16), (16,117,245), (16,20,245)]
colors_hagrid = [(245,117,16), (117,245,16), (16,117,245), (56,125,200), (88,22,152), (50,180,255), (99,22,255), (100,90,87), (64,180,120)]

# Use NGROK
#run_with_ngrok(app)


# define routes
@app.route("/")
def home():
    return render_template("home.html")

@app.route('/webcam')
def webcam():
    return render_template("webcam.html")

@app.route('/video_feed_hagrid')
def video_feed_hagrid():
    return Response(get_prediction_hagrid(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    return Response(get_prediction(), mimetype='multipart/x-mixed-replace; boundary=frame')


# define functions
def get_prediction(): 
    sequence = []
    predictions = []
    gesto = ""
    threshold = 0.5

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5) as hands:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            if not ret:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            
            # Make detections
            image, results = mediapipe_detection(frame, hands)
            print(results)        
            print("which hand?", results.multi_handedness)

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
                print(labels[np.argmax(res)])
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


# HAGRID DATASET
# capire perchè funziona molto meglio questa che quelle di sotto (generate_frames)
def get_prediction_hagrid():
    
    sequence = []
    predictions = []
    gesto = ""
      
    while cap.isOpened():
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1) # specchiamo il frame

            with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5) as hands:
                
                  # Make detections
                  image, results = mediapipe_detection(frame, hands)          

                  first_hand_keypoints = np.zeros(21*3)
                  second_hand_keypoints = np.zeros(21*3)

                  if results.multi_hand_landmarks:
                      for num, hand_landmarks in enumerate(results.multi_hand_landmarks):

                          mp_drawing.draw_landmarks(
                              image,
                              hand_landmarks,
                              mp_hands.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(150, 50, 10), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(
                                  color=(80,255,20), thickness=2, circle_radius=2)
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
                                  
                  # 2. Prediction logic
                  sequence.append(keypoints)
                  sequence = sequence[-30:]
                  
                  if len(sequence) == 30:                      
                      res = grid_search.predict_proba([keypoints])
                      res = res[0] # è contenuto in un array | TODO togliere
                      #print(labels_hagrid[np.argmax(res)])
                      predictions.append(np.argmax(res))
                      
                  #3. Viz logic
                      if np.unique(predictions[-10:])[0]==np.argmax(res): 
                          if res[np.argmax(res)] > 0.5:                     
                              gesto = labels_hagrid[np.argmax(res)]    
                              print(gesto)                          
                      
                      # Viz probabilities
                      image = prob_viz(res, labels_hagrid, image, colors_hagrid)
                      
                  cv2.rectangle(image, (0,0), (640, 40), (80, 50, 30), -1)
                  cv2.putText(image, ' '.join(gesto), (3,30), 
                                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)                  
                                   
                
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')  # concat frame one by one and show result      

# funzioni che insieme fanno lo stesso di quella sopra, ma funzionano peggio
def generate_frames():
    
    sequence = []
    predictions = []

    while True:
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1) # specchiamo il frame
            frame = predict(frame, sequence, predictions)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
def predict(frame, sequence, predictions):
    
    gesto = ""

    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5) as hands:
                
        # Make detections
        image, results = mediapipe_detection(frame, hands)          

        first_hand_keypoints = np.zeros(21*3)
        second_hand_keypoints = np.zeros(21*3)

        if results.multi_hand_landmarks:
            for num, hand_landmarks in enumerate(results.multi_hand_landmarks):

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(
                        color=(245, 66, 230), thickness=2, circle_radius=2)
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
                        
        # 2. Prediction logic
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:                      
            res = grid_search.predict_proba([keypoints])
            res = res[0] # è contenuto in un array | TODO togliere
            print(labels_hagrid[np.argmax(res)])
            predictions.append(np.argmax(res))
            
        #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > 0.5:                     
                    gesto = labels_hagrid[np.argmax(res)]
            
            # Viz probabilities
            image = prob_viz(res, labels_hagrid, image, colors_hagrid)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(gesto), (3,30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        return image


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
        
    return output_frame


if __name__ == "__main__":
    app.run(debug=True)

# TODO 1): capire come stoppare webcam
# TODO 2): estrarre il risultato del processo di riconoscimento (la prediction finale del gesto eseguito) 
