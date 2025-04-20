import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
import easyocr
import geocoder
from datetime import datetime
import pytz
import time
import telepot
from PIL import ImageFont, ImageDraw, Image
from Slogans import slogans

# EasyOCR
reader = easyocr.Reader(['bn', 'en'])

# Video Capture
# cap = cv2.VideoCapture("fight.mp4")
cap = cv2.VideoCapture("poster.mp4")
width = 720
height = 640

# Load YOLO Models
violenceDetect_model = YOLO("best.pt")
person_model = YOLO("yolo11n.pt")
classNames = ['Fight', 'Normal', 'Poster']

# Set Thresholds
PERSON_CONFIDENCE_THRESHOLD = 0.5
VIOLENCE_CONFIDENCE_THRESHOLD = 0.5

# Bot Setup
IST = pytz.timezone('Asia/dhaka')
raw_TS = datetime.now(IST)
curr_date = raw_TS.strftime("%d-%m-%y")
curr_time = raw_TS.strftime("%H:%M:%S")

bot_token = '7798758169:AAHlEV_yJKnsd9YTPN-sdR05vBjYPE06f3E'
chat_id = '-1002448159857'
bot = telepot.Bot(bot_token)

# Get Location
g = geocoder.ip('me')
latitude, longitude = g.latlng
google_maps_url = f"https://www.google.com/maps?q={latitude},{longitude}"
filename = 'savedImage.jpg'

# Counter and Timer
detection_counter = 0
start_time = time.time()

# Bangla Font
bangla_font_path = 'Helal Hafiz Bold.ttf'
font_size = 20
bangla_font = ImageFont.truetype(bangla_font_path, font_size)

# Bangla Text Overlay

def put_bangla_text(image, text, position, font, color=(255, 255, 255)):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

while True:
    success, frame = cap.read()
    if not success:
        break

    resized_frame = cv2.resize(frame, (width, height))

    # Person Detection
    person_results = person_model(resized_frame)
    persons = []

    for result in person_results:
        boxes = result.boxes
        for box in boxes:
            conf = box.conf[0]
            if conf > PERSON_CONFIDENCE_THRESHOLD and int(box.cls[0]) == 0:
                x1, y1, x2, y2 = box.xyxy[0]
                persons.append((int(x1), int(y1), int(x2), int(y2)))

    # Count Heads
    head_count = 0
    for x1, y1, x2, y2 in persons:
        head_roi = resized_frame[int(y1):int(y1 + (y2 - y1) * 0.2), int(x1):int(x2)]
        if head_roi.size > 0:
            head_count += 1
            cv2.rectangle(resized_frame, (x1, y1), (x2, int(y1 + (y2 - y1) // 2)), (255, 0, 0), 2)
            cv2.putText(resized_frame, f"Head {head_count}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Violence Detection
    if persons:
        violence_results = violenceDetect_model(resized_frame)
        for result in violence_results:
            boxes = result.boxes
            for box in boxes:
                conf = box.conf[0]
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if conf > VIOLENCE_CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if currentClass == 'Fight':
                        for px1, py1, px2, py2 in persons:
                            if (x1 < px2 and x2 > px1 and y1 < py2 and y2 > py1):
                                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cvzone.putTextRect(resized_frame, f'Fight {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                                detection_counter += 1
                                break

                    elif currentClass == 'Normal':
                        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cvzone.putTextRect(resized_frame, f'Normal {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

                    elif currentClass == 'Poster' and head_count > 10:
                        cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cvzone.putTextRect(resized_frame, f'Poster {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                        poster_crop = resized_frame[y1:y2, x1:x2]
                        result = reader.readtext(poster_crop)
                        detected_text = ' '.join([text[1] for text in result])
                        resized_frame = put_bangla_text(resized_frame, detected_text, (max(0, x1), max(35, y1 - 35)), bangla_font, color=(0, 0, 255))
                        for slogan in slogans:
                            if slogan in detected_text:
                                print("Yes")
                                detection_counter += 1
                                break
    # Display the count on the frame
    cv2.putText(resized_frame, f"Head Count: {head_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    current_time_check = time.time()
    if current_time_check - start_time >= 60:
        detection_counter = 0
        start_time = current_time_check

    if detection_counter != 0 and detection_counter % 10 == 0:
        print(f"Detection Counter: {detection_counter}")
        cv2.putText(resized_frame, f"Detection Counter: {detection_counter}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(filename, frame)
        bot.sendMessage(chat_id, f"VIOLENCE ALERT!! \nLOCATION: {google_maps_url} \nDate: {curr_date} \nTIME: {curr_time}")
        bot.sendPhoto(chat_id, photo=open(filename, 'rb'))

    cv2.imshow('Video', resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()