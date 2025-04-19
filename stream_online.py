
import subprocess
import re

from threading import Thread
from flask import Flask, Response
import cv2
app = Flask(__name__)

# Open the webcam (0 is usually the default for the first camera)
camera = cv2.VideoCapture(0)
# camera = cv2.VideoCapture("poster.mp4")

# Define desired resolution (lower resolution = less bandwidth usage)
video_width = 720
video_height = 480

# Set the camera resolution (optional: adjust to your desired lower resolution)
camera.set(3, video_width)  # Width
camera.set(4, video_height)  # Height


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
    """Run Cloudflare Tunnel and keep it open"""
    try:
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

        process.wait()

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Start the Flask app in a separate thread
    flask_thread = Thread(target=start_flask_app)
    flask_thread.start()

    # Start the Cloudflare tunnel
    start_cloudflare_tunnel()
