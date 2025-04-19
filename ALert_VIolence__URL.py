import cvzone
import numpy as np
from ultralytics import YOLO
import telepot
from datetime import datetime
import pytz
import geocoder
import subprocess
import re
from threading import Thread
from flask import Flask, Response
import cv2


app = Flask(__name__)

# For video capture
cap = cv2.VideoCapture("poster.mp4")
# cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("/content/demo1.gif")
camera = cv2.VideoCapture(0)

# Define desired resolution (lower resolution = less bandwidth usage)
video_width = 320
video_height = 240

# Set the camera resolution (optional: adjust to your desired lower resolution)
camera.set(3, video_width)  # Width
camera.set(4, video_height)  # Height
# Load YOLO models
violenceDetect_model = YOLO("best.pt")
person_model = YOLO("yolo11n.pt")
classNames = ['Fight', 'Normal', 'Poster']
# Set confidence thresholds
writer = None
(W, H) = (None, None)
PERSON_CONFIDENCE_THRESHOLD = 0.5
VIOLENCE_CONFIDENCE_THRESHOLD = 0.5

#Variables for send data using bot
#get current time and date
IST = pytz.timezone('Asia/dhaka')
raw_TS = datetime.now(IST)
curr_date=raw_TS.strftime("%d-%m-%y")
curr_time=raw_TS.strftime("%H:%M:%S")
#Bot Connectivity
bot_token='7798758169:AAHlEV_yJKnsd9YTPN-sdR05vBjYPE06f3E'
chat_id='-1002448159857'
bot = telepot.Bot(bot_token)
# Get the current location using geocoder's IP method
g = geocoder.ip('me')
latitude, longitude = g.latlng
google_maps_url = f"https://www.google.com/maps?q={latitude},{longitude}"
filename = 'savedImage.jpg'

#for Generating network url
def generate():
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        # Resize the frame to reduce video quality
        frame_resized = cv2.resize(frame, (video_width, video_height))

        # Convert the resized image to JPEG
        ret, jpeg = cv2.imencode('.jpg', frame_resized)
        if not ret:
            break
        # Yield the image in the MJPEG stream format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


def start_flask_app():
    """Start Flask app to stream video"""
    app.run(host='0.0.0.0', port=5000, debug=False)


def start_cloudflare_tunnel():
    process = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://localhost:5000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    tunnel_url = None
    # Read the output in real-time to find the generated URL
    for line in iter(process.stdout.readline, ''):
        print(line.strip())  # Print Cloudflare output live

        # Extract the Cloudflare URL using regex
        match = re.search(r"https://[a-zA-Z0-9_-]+\.trycloudflare\.com", line)
        if match and tunnel_url is None:  # Only capture the URL once
            tunnel_url = match.group(0)
            print(f"\n🔗 Live Stream URL: {tunnel_url}/video_feed\n")
            return tunnel_url

    process.wait()

if __name__ == "__main__":
    # Start the Flask app in a separate thread
    flask_thread = Thread(target=start_flask_app)
    flask_thread.start()

    # Start the Cloudflare tunnel
    start_cloudflare_tunnel()

#main section start
while True:
    success, frame = cap.read()
    resized_frame = cv2.resize(frame, (video_width, video_height))

    if not success:
        break

    if W is None or H is None:
        (H, W) = resized_frame.shape[:2]

    # Perform person detection
    person_results = person_model(resized_frame)
    persons = []

    for result in person_results:
        boxes = result.boxes
        for box in boxes:
            conf = box.conf[0]
            if conf > PERSON_CONFIDENCE_THRESHOLD and int(box.cls[0]) == 0:  # Class '0' for 'person'
                x1, y1, x2, y2 = box.xyxy[0]
                persons.append((int(x1), int(y1), int(x2), int(y2)))

    # Perform violence detection if a person is detected
    if persons:
        violence_results = violenceDetect_model(resized_frame)
        for result in violence_results:
            boxes = result.boxes
            for box in boxes:
                conf = box.conf[0]
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if conf > VIOLENCE_CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Draw rectangles based on the detection class
                    if currentClass == 'Fight':
                        print("Fight")
                        # Check if the violence box overlaps with any person box
                        for px1, py1, px2, py2 in persons:
                            if (x1 < px2 and x2 > px1 and y1 < py2 and y2 > py1):
                                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cvzone.putTextRect(resized_frame, f'Fight {conf:.2f}', (max(0, x1), max(35, y1)),
                                                   scale=1, thickness=1)
                                # # send data to bot
                                # cv2.imwrite(filename, frame)
                                # bot.sendMessage(chat_id,f"VIOLENCE ALERT!! \nLOCATION: {google_maps_url} \nDate: {curr_date} \nTIME: {curr_time}")
                                # bot.sendPhoto(chat_id, photo=open(filename, 'rb'))
                                # break

                    elif currentClass == 'Normal':
                        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cvzone.putTextRect(resized_frame, f'Normal {conf:.2f}', (max(0, x1), max(35, y1)), scale=1,
                                           thickness=1)


                    elif currentClass == 'Poster':

                        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cvzone.putTextRect(resized_frame, f'Poster {conf:.2f}', (max(0, x1), max(35, y1)), scale=1,
                                           thickness=1)

    # Show the output image
    cv2.imshow('Video', frame)
    key = cv2.waitKey(1000) & 0xFF

    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Release the file pointers
print("[INFO] cleaning up...")
writer.release()
cap.release()