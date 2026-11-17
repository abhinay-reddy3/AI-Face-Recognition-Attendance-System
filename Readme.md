# AI Face Recognition Attendance System

A contactless, real-time classroom attendance web application built with **Python**, **Flask**, **OpenCV**, and **TensorFlow (MobileNetV2)**. Designed for academic environments, the system detects multiple faces simultaneously from a live webcam stream, verifies registered students against a negative background contrast set, enforces a strict 1-hour anti-duplication lock, and logs session-tagged attendance to CSV.

---

## Key Features

* **Contactless Multi-Face Recognition:** Captures 720p HD video, detects multiple faces at classroom distance using OpenCV Haar Cascades, and processes all detected faces in a single vectorized MobileNetV2 batch.
* **Strict Student-Only Output:** Trains a custom neural classifier head on top of L2-normalized 1280-dimensional MobileNetV2 embeddings using registered student datasets alongside Kaggle benchmark faces grouped as a single `"Unknown"` background class.
* **Temporal Verification & 1-Hour Deduplication:** Requires 3 consecutive inference detections before logging attendance and enforces a thread-safe, CSV-backed 3600-second (1-hour) lock per student to prevent duplicate entries.
* **Class Session Metadata Workflow:** Includes an admin login portal (`/login`) followed by a Class Session Details selector (`/select_session`) to tag every attendance record with Semester, Section, Year, Batch, Course, Date, and Faculty Name.
* **Live Auto-Updating Dashboard:** Displays the real-time annotated video stream, active class metadata banner, auto-refreshing attendance log table, one-click log reset, and `.csv` report export.

---

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Backend & Routing** | Python 3.9+, Flask |
| **Computer Vision** | OpenCV (`opencv-python < 5.0`, Haar Cascade Classifier) |
| **Deep Learning** | TensorFlow / Keras (MobileNetV2 Backbone + Dense Softmax Head) |
| **Data & Math** | NumPy, `csv`, `pickle`, `threading` |
| **Frontend UI** | HTML5, CSS3, Vanilla JavaScript (Fetch API polling) |

---

## Project Structure

```text
AI-Face-Recognition-Attendance-System/
├── app.py                     # Main Flask server, video stream generator & attendance logger
├── train_model.py             # Feature extraction, augmentation & classifier training script
├── dataset/
│   ├── Students/              # Registered student folders (Format: name_enrollmentNo)
│   │   ├── abhinay_24STUCHH010976/
│   │   ├── yagna_24STUCHH010895/
│   │   ├── tanay_24STUCHH010674/
│   │   └── bhuvana_24STUCHH010938/
│   └── Kaggle_Faces/          # Negative contrast dataset (mapped to 'Unknown')
├── models/
│   ├── face_classifier.keras  # Saved Keras neural classifier head
│   └── class_names.pkl        # Serialized list of valid output labels
├── data/
│   └── attendance.csv         # Auto-generated attendance log with session metadata
├── static/
│   └── campus.jpg             # Dashboard background asset
└── templates/
    ├── login.html             # Admin authentication page
    ├── select_session.html    # Class metadata selection form
    └── index.html             # Live camera stream & attendance dashboard

System Pipeline
Live Webcam (720p) ──► Haar Cascade Face Detection ──► 160×160 RGB Crop & [-1, 1] Scaling
                                                                    │
CSV Attendance Log ◄── 1-Hr Deduplication Lock ◄── 3-Cycle Filter ◄─┴─ MobileNetV2 (1280-D) + L2 Norm

Setup & Installation
1. Create and Activate a Virtual Environment
macOS / Linux:
python3 -m venv venv
source venv/bin/activate

Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

2. Install Dependencies
pip install flask "opencv-python<5.0" tensorflow numpy pillow

3. Prepare the Dataset
 * Create a folder for each student inside dataset/Students/ using the naming convention name_enrollment (e.g., abhinay_24STUCHH010976).
 * Add 10–20 clear, front-facing photos per student inside their respective folder.
 * Place general/unregistered face images inside dataset/Kaggle_Faces/ to serve as negative background contrast.
Usage
Step 1: Train the Classifier
Run the training script to extract augmented MobileNetV2 embeddings and save the trained model into models/:
python3 train_model.py

Step 2: Start the Web Application
Launch the Flask server:
python3 app.py

Step 3: Access the Portal
 * Open your browser and navigate to http://127.0.0.1:5000.
 * Login: Sign in using the default admin credentials (admin / admin123).
 * Select Session: Enter the active Semester, Section, Year, Batch, Course, Date, and Faculty Name, then click Proceed to Attendance Dashboard.
 * Mark Attendance: Stand in front of the webcam. Registered students are highlighted with a green bounding box and logged automatically to data/attendance.csv, while unregistered faces appear in red as Unknown.
Team & Supervision
 * Developed By: K. Abhinay Reddy (24STUCHH010976), R. Yagna Sri (24STUCHH010895), Tanay Sharma (24STUCHH010674), T. Bhuvana (24STUCHH010938)
 * Supervised By: Dr. Ramesh Ponnala & Dr. Sowjanya Ramisitty
 * Institution: Faculty of Science & Technology (FST), The ICFAI Foundation for Higher Education, Hyderabad
