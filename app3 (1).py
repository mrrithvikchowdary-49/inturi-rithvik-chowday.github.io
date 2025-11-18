import os
import cv2
import numpy as np
from datetime import datetime
import tensorflow as tf
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# ==============================
# CONFIG — YOUR VIDEO FILE PATH
# ==============================
VIDEO_SOURCE = r"E:\CrowdProject\crowdProject\videos\violence.mp4"  # <-- YOUR PATH HERE
MODEL_PATH = r"modelnew.h5"

SENDER_EMAIL = "lokesh.s26vit@gmail.com"
RECEIVER_EMAIL = "lokesh.s26gm@gmail.com"
GMAIL_APP_PASSWORD = "phqa rdvg tzjj axyp"
##def send_email(subject, message, image, time):
##sender_email = "lokesh.s26vit@gmail.com"
##password = "phqa rdvg tzjj axyp"  # Gmail App Password
##receiver_email = "lokesh.s26gm@gmail.com"

# ==============================
# COMPATIBILITY SHIM (Fixes: groups=1)
# ==============================
class DepthwiseConv2DCompat(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)

def load_model_compat(path):
    return tf.keras.models.load_model(
        path,
        compile=False,
        custom_objects={
            "DepthwiseConv2D": DepthwiseConv2DCompat,
            "relu6": tf.nn.relu6,
        },
    )

# ==============================
# Load Face Cascade
# ==============================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ==============================
# Load Model
# ==============================
try:
    model = load_model_compat(MODEL_PATH)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(" Model loaded successfully (compat mode).")
except Exception as e:
    print(f" ERROR loading model: {e}")
    raise SystemExit

# ==============================
# Constants
# ==============================
VIOLENCE_THRESHOLD = 0.8
EMAIL_FRAME_COUNT = 5
COOLDOWN_FRAMES = 150
MOTION_THRESHOLD = 5000000

# ==============================
# Helper Functions
# ==============================
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (128, 128)).astype("float32") / 255.0
    return frame

def predict_violence(frame):
    x = preprocess_frame(frame)
    x = np.expand_dims(x, 0)
    pred = model.predict(x, verbose=0)
    confidence = float(pred[0][0])
    label = "Violence" if confidence > VIOLENCE_THRESHOLD else "Normal"
    return label, confidence

def motion_detect(prev_frame, curr_frame):
    diff = cv2.absdiff(curr_frame, prev_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return np.sum(gray)

def send_email(subject, message, image_path, time_str):
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(f"{message}\nDetected at {time_str}", 'plain'))

        with open(image_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-Disposition", "attachment", filename=os.path.basename(image_path))
            msg.attach(img)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, GMAIL_APP_PASSWORD)
            smtp.send_message(msg)

        print(" Email sent.")

    except Exception as e:
        print(f"Email error: {e}")

    try:
        os.remove(image_path)
    except:
        pass

# ==============================
# Video Capture (FILE MODE)
# ==============================
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print(f" Unable to open video: {VIDEO_SOURCE}")
    raise SystemExit

print(f"🎥 Running detection on: {VIDEO_SOURCE}")

total_violent_frames = 0
email_sent_cooldown = 0
prev_frame = None
save_dir = "images"
os.makedirs(save_dir, exist_ok=True)

# ==============================
# Main Loop
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        print(" End of video reached.")
        break

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Face Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Motion Detection
    motion_score = motion_detect(prev_frame, frame) if prev_frame is not None else 0
    prev_frame = frame.copy()

    # CNN Prediction
    label, confidence = predict_violence(frame)
    is_violent = (label == "Violence") or (motion_score > MOTION_THRESHOLD)

    # Decision Logic
    if is_violent:
        total_violent_frames += 1
        cv2.putText(frame, f"Violence! Conf: {confidence:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if total_violent_frames >= EMAIL_FRAME_COUNT and email_sent_cooldown == 0:
            print(" Violence threshold reached! Sending alert...")

            image_path = os.path.join(save_dir, f"violence_{current_time}.jpg")
            cv2.imwrite(image_path, frame)

            send_email("Violence Detected", "A violent event has been detected.", image_path, current_time)

            email_sent_cooldown = COOLDOWN_FRAMES
            total_violent_frames = 0

    else:
        total_violent_frames = 0
        cv2.putText(frame, "Normal", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if email_sent_cooldown > 0:
        email_sent_cooldown -= 1

    cv2.imshow("Violence Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
