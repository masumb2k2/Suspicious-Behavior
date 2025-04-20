from flask import Flask, Response
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)  # Use the default camera
camera = cv2.VideoCapture("poster.mp4")

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            # End of video — rewind to the start
            camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue  # Try reading again\
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "Video Streaming Server is running. Access the video at /video_feed"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
