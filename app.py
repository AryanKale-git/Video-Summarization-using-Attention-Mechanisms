#VIDEO SUMMARIZATION - A DEEP LEARNING PROJECT

from flask import Flask, render_template, request, redirect, url_for, send_file
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

def summarize_video(input_path, output_path, frame_skip=5):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) // 2  # Reduce output video FPS

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:  # Skip frames for faster processing
            continue

        mask = back_sub.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        _, thresh = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_movement = False

        for contour in contours:
            if cv2.contourArea(contour) > 1500:  # Filter based on contour size
                significant_movement = True
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Movement", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

        if significant_movement:  # Only write frames with motion
            out.write(frame)

    cap.release()
    out.release()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return redirect(url_for("index"))

    video = request.files["video"]

    # Validate video file type
    if not video.filename.endswith(('.mp4', '.avi', '.mov')):
        return "Unsupported file type", 400

    input_path = os.path.join("uploads", secure_filename(video.filename))
    output_path = os.path.join("uploads", "summary_" + secure_filename(video.filename))

    try:
        # Save the uploaded video
        video.save(input_path)

        # Summarize the video
        summarize_video(input_path, output_path)

    except Exception as e:
        return str(e), 500

    return send_file(output_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)