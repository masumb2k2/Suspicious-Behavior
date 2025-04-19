import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
import easyocr
import geocoder
from datetime import datetime
import pytz
import math
import requests
import telepot
from PIL import ImageFont, ImageDraw, Image


#EasyOCR
reader = easyocr.Reader(['bn','en'])

# For video capture
cap = cv2.VideoCapture("poster.mp4")
# Desired dimensions
width = 720
height = 640

#cap = cv2.VideoCapture("/content/demo1.gif")
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

#Bangla Font
# Load Bangla font
bangla_font_path = 'Helal Hafiz Bold.ttf'
font_size = 20
bangla_font = ImageFont.truetype(bangla_font_path, font_size)

# Function to put Bangla text on an image
def put_bangla_text(image, text, position, font, color=(255, 255, 255)):
    # Convert OpenCV image (BGR) to PIL image (RGB)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    draw.text(position, text, font=font, fill=color)
    # Convert PIL image (RGB) back to OpenCV image (BGR)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

while True:
    success, frame = cap.read()
    resized_frame = cv2.resize(frame, (width, height))

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
                                cvzone.putTextRect(resized_frame, f'Fight {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                                # # send data to bot
                                # cv2.imwrite(filename, frame)
                                # bot.sendMessage(chat_id,f"VIOLENCE ALERT!! \nLOCATION: {google_maps_url} \nDate: {curr_date} \nTIME: {curr_time}")
                                # bot.sendPhoto(chat_id, photo=open(filename, 'rb'))
                                # break

                    elif currentClass == 'Normal':
                        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cvzone.putTextRect(resized_frame, f'Normal {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1)


                    elif currentClass == 'Poster':

                        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cvzone.putTextRect(resized_frame, f'Poster {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                        # Crop the detected poster area
                        poster_crop = resized_frame[y1:y2, x1:x2]
                        # Use EasyOCR to extract text from the cropped poster
                        result = reader.readtext(poster_crop)
                        # Concatenate all detected text
                        detected_text = ' '.join([text[1] for text in result])
                        # Display the detected text on the frame
                        resized_frame = put_bangla_text(resized_frame, detected_text, (max(0, x1), max(35, y1 - 35)), bangla_font,color=(0, 0, 255))
                        print(f"Extracted Text: {detected_text}")

    # # Display the resulting frame
    # if writer is None:
    #     # initialize our video writer
    #     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    #     writer = cv2.VideoWriter("output/v_output.avi", fourcc, 30, (W, H), True)
    #
    # # Write the output frame to disk
    # writer.write(frame)

    # Show the output image
    cv2.imshow('Video', resized_frame)
    key = cv2.waitKey(1) & 0xFF

    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Release the file pointers
print("[INFO] cleaning up...")
# writer.release()
cap.release()