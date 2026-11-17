from flask import Flask, render_template, request, redirect, url_for, session, Response, send_file, jsonify
from datetime import datetime
import threading
import csv
import os
import pickle
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

app = Flask(__name__)
app.secret_key = "super_secret_attendance_key"

os.makedirs("data", exist_ok=True)
ATTENDANCE_FILE = "data/attendance.csv"
CLASSIFIER_PATH = "models/face_classifier.keras"
CLASSES_PATH = "models/class_names.pkl"

# Thread lock prevents duplicate writes when multiple browser tabs or frames fire simultaneously
attendance_lock = threading.Lock()

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml" if hasattr(cv2, 'data') else "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(160, 160, 3),
    pooling="avg"
)

classifier = None
class_names = []

if os.path.exists(CLASSIFIER_PATH) and os.path.exists(CLASSES_PATH):
    classifier = tf.keras.models.load_model(CLASSIFIER_PATH)
    with open(CLASSES_PATH, "rb") as f:
        class_names = pickle.load(f)
    print(f"[INFO] Loaded Trained Classifier. Allowed Output Labels: {class_names}")
else:
    print("[WARNING] Trained model not found! Run 'python3 train_model.py' first.")

last_logged_time = {}
temporal_counters = {}

def extract_batch_embeddings(face_crops):
    """Processes multiple detected faces simultaneously in a single vectorized batch."""
    batch_tensors = []
    for crop in face_crops:
        face_resized = cv2.resize(crop, (160, 160))
        rgb_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        batch_tensors.append(rgb_face.astype(np.float32))

    batch_arr = preprocess_input(np.array(batch_tensors, dtype=np.float32))
    raw_vectors = base_model(batch_arr, training=False).numpy()
    norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return raw_vectors / norms

def has_logged_within_hour(student_name, enrollment_no, current_dt, session_date):
    """Checks both memory cache AND data/attendance.csv to guarantee 1 entry per hour."""
    person_key = f"{student_name.strip().upper()}_{enrollment_no.strip().upper()}"

    # 1. Fast in-memory check
    if person_key in last_logged_time:
        if (current_dt - last_logged_time[person_key]).total_seconds() < 3600:
            return True

    # 2. Persistent CSV check (survives server restarts & page reloads)
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, mode='r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_name = row.get("Name", "").strip().upper()
                row_enr = row.get("Enrollment No", "").strip().upper()
                row_date = row.get("Date", "").strip()
                row_time = row.get("Time", "").strip()

                if (row_name == student_name.strip().upper() or row_enr == enrollment_no.strip().upper()) and row_date == session_date:
                    try:
                        logged_dt = datetime.strptime(f"{row_date} {row_time}", "%Y-%m-%d %H:%M:%S")
                        if abs((current_dt - logged_dt).total_seconds()) < 3600:
                            last_logged_time[person_key] = logged_dt
                            return True
                    except ValueError:
                        continue
    return False

def mark_attendance(student_name, enrollment_no, session_data):
    """Thread-safe attendance logger enforcing strictly 1 entry per student per hour."""
    with attendance_lock:
        now = datetime.now()
        session_date = session_data.get('session_date', now.strftime("%Y-%m-%d"))

        # Block duplicate if already marked within 3600 seconds (1 hour)
        if has_logged_within_hour(student_name, enrollment_no, now, session_date):
            return False

        file_exists = os.path.isfile(ATTENDANCE_FILE)
        semester = session_data.get('semester', 'N/A')
        section = session_data.get('section', 'N/A')
        batch = session_data.get('batch', 'N/A')
        course = session_data.get('course', 'N/A')
        year = session_data.get('year', 'N/A')
        faculty_name = session_data.get('faculty_name', 'N/A')
        timestamp = now.strftime("%H:%M:%S")

        with open(ATTENDANCE_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists or os.stat(ATTENDANCE_FILE).st_size == 0:
                writer.writerow(["Name", "Enrollment No", "Date", "Time", "Semester", "Section", "Batch", "Course", "Year", "Faculty", "Status"])
            writer.writerow([student_name, enrollment_no, session_date, timestamp, semester, section, batch, course, year, faculty_name, "Present"])

        person_key = f"{student_name.strip().upper()}_{enrollment_no.strip().upper()}"
        last_logged_time[person_key] = now
        print(f"[SUCCESS] Attendance Marked (1-Hr Lock Active): {student_name} ({enrollment_no}) at {timestamp}")
        return True

def get_attendance_records():
    records = []
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
    return list(reversed(records))

def gen_frames(session_data):
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_count = 0
    cached_detections = []

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            frame_count += 1

            if frame_count % 4 == 0 and classifier is not None:
                cached_detections = []
                small_gray = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
                
                faces_small = face_cascade.detectMultiScale(
                    small_gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(22, 22)
                )

                valid_boxes = []
                face_crops = []
                img_h, img_w = frame.shape[:2]

                for (sx, sy, sw, sh) in faces_small:
                    x, y, w, h = sx * 2, sy * 2, sw * 2, sh * 2
                    pad_w, pad_h = int(w * 0.1), int(h * 0.1)
                    x1 = max(0, x - pad_w)
                    y1 = max(0, y - pad_h)
                    x2 = min(img_w, x + w + pad_w)
                    y2 = min(img_h, y + h + pad_h)

                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        valid_boxes.append((x, y, w, h))
                        face_crops.append(crop)

                if len(face_crops) > 0:
                    live_vectors = extract_batch_embeddings(face_crops)
                    all_probs = classifier(live_vectors, training=False).numpy()

                    seen_keys_in_frame = set()

                    for ((x, y, w, h), probs) in zip(valid_boxes, all_probs):
                        best_idx = int(np.argmax(probs))
                        confidence = float(probs[best_idx])
                        predicted_label = class_names[best_idx]

                        if predicted_label != "Unknown" and confidence >= 0.55:
                            parts = predicted_label.split('_')
                            student_name = parts[0].strip().upper()
                            enrollment_no = parts[1].strip().upper() if len(parts) > 1 else student_name
                            person_key = f"{student_name}_{enrollment_no}"

                            seen_keys_in_frame.add(person_key)
                            color = (0, 255, 0)

                            temporal_counters[person_key] = temporal_counters.get(person_key, 0) + 1

                            if temporal_counters[person_key] >= 3:
                                mark_attendance(student_name, enrollment_no, session_data)

                            label = f"{student_name} ({confidence*100:.1f}%)"
                        else:
                            color = (0, 0, 255)
                            label = "Unknown"

                        cached_detections.append((x, y, w, h, label, color))

                    for key in list(temporal_counters.keys()):
                        if key not in seen_keys_in_frame:
                            temporal_counters[key] = 0

            for (x, y, w, h, label, color) in cached_detections:
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.rectangle(frame, (x, max(0, y-32)), (x+w, y), color, cv2.FILLED)
                cv2.putText(frame, label, (x+6, max(20, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            display_frame = cv2.resize(frame, (854, 480))
            ret, buffer = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        camera.release()

@app.route('/')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    if 'semester' not in session:
        return redirect(url_for('select_session'))
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'admin123':
            session['user'] = username
            return redirect(url_for('select_session'))
        else:
            return render_template('login.html', error="Invalid Credentials")
    return render_template('login.html')

@app.route('/select_session', methods=['GET', 'POST'])
def select_session():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        session['semester'] = request.form.get('semester')
        session['section'] = request.form.get('section')
        session['batch'] = request.form.get('batch')
        session['course'] = request.form.get('course')
        session['year'] = request.form.get('year')
        session['session_date'] = request.form.get('session_date')
        session['faculty_name'] = request.form.get('faculty_name')
        return redirect(url_for('dashboard'))

    today_date = datetime.now().strftime("%Y-%m-%d")
    return render_template('select_session.html', today_date=today_date)

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    if 'semester' not in session:
        return redirect(url_for('select_session'))
        
    session_info = {
        'semester': session.get('semester'),
        'section': session.get('section'),
        'batch': session.get('batch'),
        'course': session.get('course'),
        'year': session.get('year'),
        'session_date': session.get('session_date'),
        'faculty_name': session.get('faculty_name')
    }
    
    records = get_attendance_records()
    return render_template('index.html', session_info=session_info, records=records)

@app.route('/api/records')
def api_records():
    if 'user' not in session:
        return jsonify([])
    return jsonify(get_attendance_records())

@app.route('/video_feed')
def video_feed():
    if 'user' not in session:
        return redirect(url_for('login'))
        
    session_data = {
        'semester': session.get('semester', 'N/A'),
        'section': session.get('section', 'N/A'),
        'batch': session.get('batch', 'N/A'),
        'course': session.get('course', 'N/A'),
        'year': session.get('year', 'N/A'),
        'session_date': session.get('session_date', datetime.now().strftime("%Y-%m-%d")),
        'faculty_name': session.get('faculty_name', 'N/A')
    }
    
    return Response(gen_frames(session_data), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download/attendance')
def download_attendance():
    if 'user' not in session:
        return redirect(url_for('login'))
    if os.path.exists(ATTENDANCE_FILE):
        return send_file(ATTENDANCE_FILE, as_attachment=True, download_name="attendance.csv")
    return "No attendance file found.", 404

@app.route('/clear_attendance', methods=['POST'])
def clear_attendance():
    if 'user' not in session:
        return redirect(url_for('login'))
    with attendance_lock:
        last_logged_time.clear()
        temporal_counters.clear()
        if os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Enrollment No", "Date", "Time", "Semester", "Section", "Batch", "Course", "Year", "Faculty", "Status"])
    return redirect(url_for('dashboard'))

@app.route('/change_session')
def change_session():
    session.pop('semester', None)
    return redirect(url_for('select_session'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    print("Starting AI Face Recognition Server...")
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)