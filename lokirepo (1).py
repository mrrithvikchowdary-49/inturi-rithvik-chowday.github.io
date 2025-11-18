import cv2
import numpy as np
from datetime import datetime
import tensorflow as tf
from keras.layers import DepthwiseConv2D
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText    
from email.mime.image import MIMEImage
import os

# ==============================
#  Patch DepthwiseConv2D to ignore unsupported 'groups' argument
# ==============================
#  Patch DepthwiseConv2D to ignore unsupported 'groups' argument
orig_init = DepthwiseConv2D.__init__

def patched_init(self, *args, **kwargs):
    kwargs.pop('groups', None)
    orig_init(self, *args, **kwargs)

DepthwiseConv2D.__init__ = patched_init
print(" Patched DepthwiseConv2D to ignore 'groups' argument.")


# ==============================
# Load face detector
# ==============================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ==============================
# Load violence detection model (single-frame CNN)
# ==============================
try:
    model = tf.keras.models.load_model(r'modelnew.h5', compile=False)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Model loaded successfully!")
except Exception as e:
    print(f"FATAL ERROR: Unable to load model. Details: {e}")
    exit()

# ==============================
# Constants
# ==============================
SEQUENCE_LENGTH = 10
VIOLENCE_THRESHOLD = 0.8
EMAIL_FRAME_COUNT = 5          # Consecutive violent frames to trigger email
COOLDOWN_FRAMES = 150          # ~5 seconds
MOTION_THRESHOLD = 5000000     # Adjust based on lighting / webcam

# ==============================
# Helper Functions
# ==============================
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (128, 128)).astype("float32")
    frame /= 255.0
    return frame

def predict_violence(frame):
    """Single-frame CNN prediction"""
    frame = preprocess_frame(frame)
    frame = np.expand_dims(frame, 0)  # Shape: (1,128,128,3)
    prediction = model.predict(frame, verbose=0)
    confidence = float(prediction[0][0])
    label = "Violence" if confidence > VIOLENCE_THRESHOLD else "Normal"
    return label, confidence

def motion_detect(prev_frame, curr_frame):
    """Detect motion between two frames"""
    diff = cv2.absdiff(curr_frame, prev_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    motion_score = np.sum(gray_diff)
    return motion_score

def send_email(subject, message, image, time):
    sender_email = "lokesh.s26vit@gmail.com"
    password = "phqa rdvg tzjj axyp"  # Gmail App Password
    receiver_email = "lokesh.s26gm@gmail.com"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(f"{message}\nDetected at {time}", 'plain'))

    try:
        with open(image, 'rb') as fp:
            img = MIMEImage(fp.read())
            img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image))
            msg.attach(img)
        print(" Image attached successfully.")

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, password)
            print("Logged in successfully.")
            smtp.send_message(msg)
            print("Email sent successfully!")

    except Exception as e:
        print(f" Failed to send email: {e}")

    try:
        os.remove(image)
    except OSError:
        pass




# ==============================
# Main Video Capture
# ==============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_sequence = []
total_violent_frames = 0
email_sent_cooldown = 0
save_dir = 'images'
os.makedirs(save_dir, exist_ok=True)

prev_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Motion detection
    motion_score = 0
    if prev_frame is not None:
        motion_score = motion_detect(prev_frame, frame)
    prev_frame = frame.copy()

    # Single-frame CNN prediction
    label, confidence = predict_violence(frame)

    # Combine motion + CNN logic for violence
    is_violent = (label == "Violence") or (motion_score > MOTION_THRESHOLD)

    # Consecutive violent frame counting & email
    if is_violent:
        total_violent_frames += 1
        box_color = (0, 0, 255)
        cv2.putText(frame, f"Violence! Conf: {confidence:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)

        if total_violent_frames >= EMAIL_FRAME_COUNT and email_sent_cooldown == 0:
            print("\n Violence Threshold Reached. Sending Email...")
            image_path = os.path.join(save_dir, f"violence_{current_time}.jpg")
            cv2.imwrite(image_path, frame)
            send_email("ALERT: Violence Detected", "Violence event detected.", image_path, current_time)
            total_violent_frames = 0
            email_sent_cooldown = COOLDOWN_FRAMES
    else:
        total_violent_frames = 0
        cv2.putText(frame, "Normal", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Cooldown update
    if email_sent_cooldown > 0:
        email_sent_cooldown -= 1

    # Overlay info
    cv2.putText(frame, "Face Detection & Crowd Monitoring", (int(frame.shape[1]/2)-300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (128,0,128), 2)
    cv2.putText(frame, current_time, (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()