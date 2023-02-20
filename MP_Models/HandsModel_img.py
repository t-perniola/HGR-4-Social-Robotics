import cv2
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

DATADIR = r'D:\TMS\dataset\egogesture\gestures\23\test\subject02\11'

# For static images:
with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
  path = os.path.join(DATADIR, '000570.jpg')  # create path (...\Color\rgb1)
  # Read an image, flip it around y-axis for correct handedness output (see
  # above).
  image = cv2.imread(path)
  # Convert the BGR image to RGB before processing.
  results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

  # Print handedness and draw hand landmarks on the image.
  print('Handedness:', results.multi_handedness)
  if not results.multi_hand_landmarks:
    print("dadaada")
  image_height, image_width, _ = image.shape
  annotated_image = image.copy()
  for hand_landmarks in results.multi_hand_landmarks:
    print('hand_landmarks:', hand_landmarks)
    print(
        f'Index finger tip coordinates: (',
        f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
        f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
    )
    mp_drawing.draw_landmarks(
        annotated_image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())

  # Draw hand world landmarks.
  if not results.multi_hand_world_landmarks:
    print("faffafa")
  for hand_world_landmarks in results.multi_hand_world_landmarks:
    mp_drawing.plot_landmarks(
      hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
