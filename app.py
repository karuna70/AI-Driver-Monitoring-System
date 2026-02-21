from flask import Flask, render_template, request
import os
import cv2
import mediapipe as mp

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_points):
    points = [landmarks[i] for i in eye_points]
    vertical1 = abs(points[1].y - points[5].y)
    vertical2 = abs(points[2].y - points[4].y)
    horizontal = abs(points[0].x - points[3].x)
    return (vertical1 + vertical2) / (2.0 * horizontal)

@app.route("/", methods=["GET", "POST"])
def index():
    processed_image = None

    if request.method == "POST":
        file = request.files["file"]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        image = cv2.imread(filepath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0

            if ear < 0.20:
                cv2.putText(image, "DROWSINESS DETECTED",
                            (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255), 3)

        cv2.imwrite(filepath, image)
        processed_image = filepath

    return render_template("index.html", image=processed_image)

if __name__ == "__main__":
   import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)