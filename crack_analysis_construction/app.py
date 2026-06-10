from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from datetime import datetime
import cv2
import matplotlib
import numpy as np
import os
import sqlite3 as sql
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    transforms = None
    Image = None
    TORCH_AVAILABLE = False

matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "crackdetect"
DB_PATH = os.path.join(app.root_path, "users.db")
UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads", "original")
PREDICTED_FOLDER = os.path.join(app.root_path, "static", "uploads", "predicted")
CHART_FOLDER = os.path.join(app.root_path, "static", "charts")
PROFILE_FOLDER = os.path.join(app.root_path, "static", "uploads", "profiles")
MODEL_FOLDER = os.path.join(app.root_path, "models")
CRACK_DATASET_FOLDER = os.path.join(app.root_path, "CD")
UNCRACKED_DATASET_FOLDER = os.path.join(app.root_path, "UD")


# ALLOWED IMAGE TYPES
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
PROJECT_FIELDS = [
    "pcat", "pname", "ptype", "duration", "area", "client", "location",
    "bheight", "ftype", "fmaterial", "pspace", "erate", "sdesc",
    "poverview", "status", "img"
]


def current_user_id():
    return session.get("user_id")


# CHECK IMAGE EXTENSION
def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def db_query(query, params=(), one=False):
    with sql.connect(DB_PATH) as con:
        con.row_factory = sql.Row
        cur = con.cursor()
        cur.execute(query, params)
        return cur.fetchone() if one else cur.fetchall()


def db_execute(query, params=()):
    with sql.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute(query, params)
        con.commit()


def require_login(message="Please login first."):
    if current_user_id():
        return None

    flash(message, "error")
    return redirect(url_for("login"))


def get_user_predictions(limit=None):
    limit_sql = " LIMIT ?" if limit else ""
    params = (current_user_id(), limit) if limit else (current_user_id(),)
    return db_query(
        f"""
        SELECT *
        FROM prediction
        WHERE userid=?
        ORDER BY id DESC{limit_sql}
        """,
        params,
        one=bool(limit)
    )


# CREATE DATABASE TABLE
def init_db():
    for folder in [
        UPLOAD_FOLDER,
        PREDICTED_FOLDER,
        CHART_FOLDER,
        PROFILE_FOLDER,
        MODEL_FOLDER,
        CRACK_DATASET_FOLDER,
        UNCRACKED_DATASET_FOLDER
    ]:
        os.makedirs(folder, exist_ok=True)

    with sql.connect(DB_PATH) as con:

        cur = con.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password TEXT,
                role INTEGER,
                name TEXT,
                dob TEXT,
                email TEXT,
                contact TEXT,
                address TEXT,
                gender TEXT,
                img TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS prediction (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                userid INTEGER,
                pred_result TEXT,
                pred_image TEXT,
                orginal_image TEXT,
                avg_depths TEXT,
                confidence REAL,
                date_time TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pcat TEXT,
                pname TEXT,
                ptype TEXT,
                duration TEXT,
                area TEXT,
                client TEXT,
                location TEXT,
                bheight TEXT,
                ftype TEXT,
                fmaterial TEXT,
                pspace TEXT,
                erate TEXT,
                sdesc TEXT,
                poverview TEXT,
                status TEXT,
                img TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                user_id INTEGER,
                review_text TEXT,
                date_time TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS engineers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                role TEXT,
                experience TEXT,
                speciality TEXT,
                contact TEXT
            )
        """)

        project_count = cur.execute("SELECT COUNT(*) FROM projects").fetchone()[0]

        if project_count == 0:

            cur.executemany(
                """
                INSERT INTO projects (
                    pcat, pname, ptype, duration, area, client, location,
                    bheight, ftype, fmaterial, pspace, erate, sdesc,
                    poverview, status, img
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        "Residential",
                        "Green Heights Apartment",
                        "Multi-storey Building",
                        "18 Months",
                        "42,000 sq.ft",
                        "Urban Living Developers",
                        "Coimbatore",
                        "12 Floors",
                        "Raft Foundation",
                        "Reinforced Concrete",
                        "Parking, lift, garden",
                        "92%",
                        "Smart residential construction with safety monitoring.",
                        "Concrete quality and structural crack inspection supported by AI analysis.",
                        "Ongoing",
                        ""
                    ),
                    (
                        "Commercial",
                        "Metro Business Complex",
                        "Commercial Tower",
                        "24 Months",
                        "65,000 sq.ft",
                        "Metro Infra Group",
                        "Chennai",
                        "15 Floors",
                        "Pile Foundation",
                        "Steel and Concrete",
                        "Office space, lobby, service floor",
                        "88%",
                        "Commercial project focused on durability and maintenance planning.",
                        "AI crack detection is used for faster inspection reporting.",
                        "Inspection",
                        ""
                    ),
                    (
                        "Infrastructure",
                        "River Link Bridge",
                        "Bridge Project",
                        "30 Months",
                        "1.8 km",
                        "Public Works Department",
                        "Madurai",
                        "32 m Piers",
                        "Deep Foundation",
                        "Prestressed Concrete",
                        "Pedestrian lane, drainage",
                        "95%",
                        "Bridge construction project requiring regular crack monitoring.",
                        "Surface images can be uploaded to detect possible structural cracks.",
                        "Completed",
                        ""
                    )
                ]
            )

        con.commit()


# INITIALIZE DATABASE
init_db()



# TRAINED SIR MODEL SUPPORT
class CrackClassifier(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 2)
        )

    def forward(self, x):
        return self.net(x)


class FastUNet(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.enc1 = nn.Conv2d(3, 16, 3, padding=1)
        self.enc2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dec1 = nn.Conv2d(32, 16, 3, padding=1)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = torch.relu(self.enc1(x))
        e2 = torch.relu(self.enc2(self.pool(e1)))
        d1 = torch.nn.functional.interpolate(e2, scale_factor=2)
        d2 = torch.relu(self.dec1(d1))
        return torch.sigmoid(self.out(d2))


ML_CLASSIFIER = None
ML_DEPTH_MODEL = None
ML_TRANSFORM = None
ML_LOAD_ERROR = None


def get_ml_models():
    global ML_CLASSIFIER, ML_DEPTH_MODEL, ML_TRANSFORM, ML_LOAD_ERROR

    if ML_CLASSIFIER is not None and ML_DEPTH_MODEL is not None:
        return ML_CLASSIFIER, ML_DEPTH_MODEL, ML_TRANSFORM

    if not TORCH_AVAILABLE:
        ML_LOAD_ERROR = "PyTorch is not installed in this Python environment."
        return None, None, None

    classifier_path = os.path.join(MODEL_FOLDER, "sdnet_classifier.pth")
    depth_path = os.path.join(MODEL_FOLDER, "sdnet_fast_depth_model.pth")

    if not os.path.exists(classifier_path) or not os.path.exists(depth_path):
        ML_LOAD_ERROR = "Trained model files are missing from the models folder."
        return None, None, None

    try:
        classifier = CrackClassifier()
        classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))
        classifier.eval()

        depth_model = FastUNet()
        depth_model.load_state_dict(torch.load(depth_path, map_location="cpu"))
        depth_model.eval()

        ML_TRANSFORM = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        ML_CLASSIFIER = classifier
        ML_DEPTH_MODEL = depth_model
        ML_LOAD_ERROR = None
    except Exception as exc:
        ML_LOAD_ERROR = str(exc)
        return None, None, None

    return ML_CLASSIFIER, ML_DEPTH_MODEL, ML_TRANSFORM


def future_depth(depth, years, alpha=0.06):
    return np.clip(depth * (1 + alpha * years), 0, 1)




def validate_surface_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return False, "Uploaded image could not be processed."

    image = cv2.resize(image, (320, 240))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    white_background_ratio = np.mean((value > 235) & (saturation < 35))
    colorful_object_ratio = np.mean(saturation > 95)

    if white_background_ratio > 0.30:
        return False, "Upload a close-up concrete, wall, road, bridge, or construction surface image. Product photos and white-background images are not valid for crack analysis."

    if colorful_object_ratio > 0.45:
        return False, "This image does not look like a construction surface. Upload a real concrete/structure surface photo for crack prediction."

    return True, ""
def run_trained_model_prediction(original_path, predicted_path):
    classifier, depth_model, transform = get_ml_models()
    if classifier is None or depth_model is None:
        raise RuntimeError(ML_LOAD_ERROR or "Trained model is not available.")

    pil_image = Image.open(original_path).convert("RGB")
    tensor = transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        logits = classifier(tensor)
        probabilities = torch.softmax(logits, dim=1)
        pred_class = probabilities.argmax(1).item()
        confidence = float(probabilities.max().item())

    label = "CRACKED" if pred_class == 1 else "UNCRACKED"
    status = "Crack Detected" if label == "CRACKED" else "No Crack Detected"

    original = cv2.imread(original_path)
    if original is None:
        raise ValueError("Uploaded image could not be processed.")
    original = cv2.resize(original, (640, 480))

    avg_depth = 0.0
    max_depth = 0.0
    if label == "CRACKED":
        with torch.no_grad():
            depth_now = depth_model(tensor)[0, 0].numpy()

        avg_depth = float(depth_now.mean())
        max_depth = float(depth_now.max())
        depth_resized = cv2.resize(depth_now, (640, 480))
        depth_uint8 = np.uint8(np.clip(depth_resized * 255, 0, 255))
        heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        highlighted = cv2.addWeighted(original, 0.65, heatmap, 0.35, 0)

        if confidence >= 0.85 or avg_depth >= 0.45:
            severity = "High Risk"
        elif confidence >= 0.65 or avg_depth >= 0.30:
            severity = "Medium Risk"
        else:
            severity = "Low Risk"
    else:
        highlighted = original
        severity = "Low Risk"

    cv2.imwrite(predicted_path, highlighted)

    log_details = (
        "[ANALYSIS] Crack classification completed.\n"
        "[ANALYSIS] Depth score estimated.\n"
        f"[ANALYSIS] Class: {label}. Confidence: {round(confidence * 100, 2)}%.\n"
        f"[ANALYSIS] Average depth score: {avg_depth:.4f}. Maximum depth score: {max_depth:.4f}."
    )

    return status, round(confidence * 100, 2), severity, log_details


def predict_crack_result(saved_path, predicted_path):
    try:
        return run_trained_model_prediction(saved_path, predicted_path)
    except Exception as model_error:
        status, confidence, severity, log_details = analyze_crack_image(
            saved_path,
            predicted_path
        )
        return (
            status,
            confidence,
            severity,
            f"[ANALYSIS NOTE] Advanced model unavailable: {model_error}\n{log_details}"
        )


def save_prediction_result(saved_filename, predicted_filename, status, confidence, severity):
    original_image = f"static/uploads/original/{saved_filename}"
    predicted_image = f"static/uploads/predicted/{predicted_filename}"
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    db_execute(
        """
        INSERT INTO prediction (
            userid, pred_result, pred_image, orginal_image,
            avg_depths, confidence, date_time
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            current_user_id(),
            status,
            predicted_image,
            original_image,
            severity,
            confidence,
            date_time
        )
    )


# SIMPLE OPENCV CRACK DETECTION
def analyze_crack_image(original_path, predicted_path):

    image = cv2.imread(original_path)

    if image is None:

        raise ValueError("Uploaded image could not be processed.")

    image = cv2.resize(image, (640, 480))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 60, 160)
    kernel = np.ones((2, 2), np.uint8)
    crack_mask = cv2.dilate(edges, kernel, iterations=1)

    crack_pixels = cv2.countNonZero(crack_mask)
    total_pixels = crack_mask.shape[0] * crack_mask.shape[1]
    crack_ratio = crack_pixels / total_pixels

    highlighted = image.copy()
    highlighted[crack_mask > 0] = [0, 0, 255]

    cv2.imwrite(predicted_path, highlighted)

    confidence = round(min(98.0, max(25.0, crack_ratio * 3200)), 2)

    if confidence >= 75:
        status = "Crack Detected"
        severity = "High Risk"
    elif confidence >= 50:
        status = "Possible Crack Detected"
        severity = "Medium Risk"
    else:
        status = "No Major Crack Detected"
        severity = "Low Risk"

    log_details = (
        "[ANALYSIS] Image prepared for surface inspection.\n"
        "[ANALYSIS] Crack-region scan completed.\n"
        f"[ANALYSIS] Crack-region ratio: {round(crack_ratio * 100, 3)}%."
    )

    return status, confidence, severity, log_details


def build_analytics_charts(predictions, user_id):

    chart_time = datetime.now().strftime("%Y%m%d%H%M%S")
    severity_chart = f"charts/severity_{user_id}_{chart_time}.png"
    confidence_chart = f"charts/confidence_{user_id}_{chart_time}.png"
    severity_path = os.path.join(CHART_FOLDER, f"severity_{user_id}_{chart_time}.png")
    confidence_path = os.path.join(CHART_FOLDER, f"confidence_{user_id}_{chart_time}.png")

    severities = [row["avg_depths"] or "Unknown" for row in predictions]
    severity_counts = {
        "Low Risk": severities.count("Low Risk"),
        "Medium Risk": severities.count("Medium Risk"),
        "High Risk": severities.count("High Risk")
    }

    plt.figure(figsize=(6, 4))
    plt.bar(
        severity_counts.keys(),
        severity_counts.values(),
        color=["#10b981", "#f59e0b", "#ef4444"]
    )
    plt.title("Severity Count")
    plt.xlabel("Severity")
    plt.ylabel("Predictions")
    plt.tight_layout()
    plt.savefig(severity_path)
    plt.close()

    dates = [row["date_time"] for row in reversed(predictions)]
    confidence = [row["confidence"] or 0 for row in reversed(predictions)]
    labels = [str(index + 1) for index in range(len(dates))]

    plt.figure(figsize=(7, 4))
    plt.plot(labels, confidence, marker="o", color="#2563eb")
    plt.title("Confidence History")
    plt.xlabel("Prediction Number")
    plt.ylabel("Confidence %")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(confidence_path)
    plt.close()

    return severity_chart, confidence_chart


def estimate_depth_forecast(prediction):
    image_path = os.path.join(
        app.root_path,
        prediction["orginal_image"].replace("/", os.sep)
    )

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Saved image could not be measured.")

    image = cv2.resize(image, (640, 480))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 160)
    kernel = np.ones((2, 2), np.uint8)
    crack_mask = cv2.dilate(edges, kernel, iterations=1)

    crack_pixels = cv2.countNonZero(crack_mask)
    total_pixels = crack_mask.shape[0] * crack_mask.shape[1]
    coverage_percent = round((crack_pixels / total_pixels) * 100, 3)

    component_count = 0
    largest_component_pixels = 0
    if crack_pixels:
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(crack_mask)
        component_count = max(component_count - 1, 0)
        if component_count:
            largest_component_pixels = int(stats[1:, cv2.CC_STAT_AREA].max())

    distance = cv2.distanceTransform(crack_mask, cv2.DIST_L2, 5)
    max_crack_width_px = round(float(distance.max() * 2), 2) if crack_pixels else 0

    if max_crack_width_px >= 10:
        measurement_risk = "High image-measured crack width"
    elif max_crack_width_px >= 5:
        measurement_risk = "Medium image-measured crack width"
    elif max_crack_width_px > 0:
        measurement_risk = "Low image-measured crack width"
    else:
        measurement_risk = "No measurable crack width"

    depth_available = False
    current_depth_score = 0.0
    future_rows = []
    depth_message = "Depth score is generated from the trained crack-depth model."
    future_message = "Predicted risk-score growth over time."

    is_cracked = str(prediction["pred_result"] or "").lower().startswith("crack")
    if is_cracked:
        try:
            classifier, depth_model, transform = get_ml_models()
            if depth_model is not None:
                pil_image = Image.open(image_path).convert("RGB")
                tensor = transform(pil_image).unsqueeze(0)
                with torch.no_grad():
                    depth_now = depth_model(tensor)[0, 0].numpy()

                depth_available = True
                current_depth_score = round(float(depth_now.mean()), 4)
                for years in [1, 3, 5, 10]:
                    future_score = round(float(future_depth(depth_now, years).mean()), 4)
                    future_rows.append({
                        "years": years,
                        "score": future_score
                    })
        except Exception as exc:
            depth_message = f"Depth model could not run: {exc}"

    return {
        "id": prediction["id"],
        "status": prediction["pred_result"],
        "severity": prediction["avg_depths"] or "Unknown",
        "confidence": float(prediction["confidence"] or 0),
        "date_time": prediction["date_time"],
        "original_image": prediction["orginal_image"],
        "predicted_image": prediction["pred_image"],
        "crack_pixels": crack_pixels,
        "coverage_percent": coverage_percent,
        "component_count": component_count,
        "largest_component_pixels": largest_component_pixels,
        "max_crack_width_px": max_crack_width_px,
        "measurement_risk": measurement_risk,
        "depth_available": depth_available,
        "current_depth_score": current_depth_score,
        "depth_message": depth_message,
        "future_available": bool(future_rows),
        "future_message": future_message,
        "forecast": future_rows
    }



# USER SESSION FOR TEMPLATES
@app.context_processor
def inject_user():
    current_profile = None

    if current_user_id():

        try:
            current_profile = db_query(
                "SELECT id, username, name, img FROM users WHERE id=?",
                (current_user_id(),),
                one=True
            )
        except Exception:
            current_profile = None

    return {
        "current_user": session.get("user"),
        "current_profile": current_profile
    }


# HOME PAGE
@app.route("/")
def home():
    projects = db_query("SELECT * FROM projects ORDER BY id DESC LIMIT 3")

    return render_template("constructor.html", projects=projects)


# CRACK DETECTION DASHBOARD
@app.route("/detect")
@app.route("/dashboard")
def detect():
    return render_template("index.html")


@app.route("/crack")
def crack():
    return render_template("crack.html")


@app.route("/crack-result")
def crack_result():

    login_redirect = require_login()
    if login_redirect:
        return login_redirect

    latest_prediction = get_user_predictions(limit=1)

    depth_forecast = estimate_depth_forecast(latest_prediction) if latest_prediction else None

    return render_template(
        "crack_result.html",
        prediction=latest_prediction,
        depth_forecast=depth_forecast
    )


# PROJECTS PAGE
@app.route("/projects")
def projects():
    project_list = db_query("SELECT * FROM projects ORDER BY id DESC")

    return render_template("projects.html", projects=project_list)


@app.route("/projects/add", methods=["GET", "POST"])
def add_project():

    if request.method == "POST":

        project_data = {
            field: request.form.get(field, "").strip()
            for field in PROJECT_FIELDS
        }
        project_data["status"] = project_data["status"] or "Planning"
        project_data["img"] = ""

        if not project_data["pname"] or not project_data["client"] or not project_data["location"]:

            flash("Please enter project name, client, and location.", "error")

            return render_template("add-project.html", project=project_data)

        db_execute(
            f"""
            INSERT INTO projects ({", ".join(PROJECT_FIELDS)})
            VALUES ({", ".join(["?"] * len(PROJECT_FIELDS))})
            """,
            tuple(project_data[field] for field in PROJECT_FIELDS)
        )

        flash("Project added successfully.", "success")

        return redirect(url_for("projects"))

    return render_template("add-project.html", project={})


@app.route("/project-details/<int:project_id>")
def project_details(project_id):
    project = db_query(
        "SELECT * FROM projects WHERE id=?",
        (project_id,),
        one=True
    )

    if not project:

        flash("Project not found.", "error")

        return redirect(url_for("projects"))

    return render_template("project-details.html", project=project)


@app.route("/customers")
def customers():
    customer_list = db_query(
        """
        SELECT id, name, email, contact, address, username
        FROM users
        ORDER BY id DESC
        """
    )

    return render_template("customers.html", customers=customer_list)


@app.route("/services")
def services():
    return render_template("services.html")


@app.route("/starter-page")
def starter_page():
    return render_template("starter-page.html")


@app.route("/script")
def script_notes():
    return render_template("script.html")


@app.route("/style")
def style_notes():
    return render_template("style.html")


# ENGINEERS PAGE
@app.route("/engineers", methods=["GET", "POST"])
def engineers():

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        role = request.form.get("role", "").strip()
        experience = request.form.get("experience", "").strip()
        speciality = request.form.get("speciality", "").strip()
        contact = request.form.get("contact", "").strip()

        if name and role:
            db_execute(
                """
                INSERT INTO engineers(name, role, experience, speciality, contact)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, role, experience, speciality, contact)
            )

            flash("Engineer added successfully.", "success")

        return redirect(url_for("engineers"))

    engineers_list = db_query("SELECT * FROM engineers ORDER BY id DESC")

    return render_template("engineers.html", engineers=engineers_list)


@app.route("/engineers/delete/<int:engineer_id>", methods=["POST"])
def delete_engineer(engineer_id):
    db_execute("DELETE FROM engineers WHERE id=?", (engineer_id,))

    flash("Engineer removed successfully.", "success")

    return redirect(url_for("engineers"))


# REVIEWS PAGE
@app.route("/reviews", methods=["GET", "POST"])
def reviews():

    if request.method == "POST":

        login_redirect = require_login("Please login first to add a review.")
        if login_redirect:
            return login_redirect

        project_id = request.form.get("project_id")
        review_text = request.form.get("review_text", "").strip()
        date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if review_text:
            db_execute(
                """
                INSERT INTO reviews(project_id, user_id, review_text, date_time)
                VALUES (?, ?, ?, ?)
                """,
                (project_id, current_user_id(), review_text, date_time)
            )

            flash("Review added successfully.", "success")

        return redirect(url_for("reviews"))

    project_list = db_query("SELECT id, pname FROM projects ORDER BY pname")
    review_list = db_query(
        """
        SELECT reviews.*, users.name, users.username, projects.pname
        FROM reviews
        LEFT JOIN users ON reviews.user_id = users.id
        LEFT JOIN projects ON reviews.project_id = projects.id
        ORDER BY reviews.id DESC
        """
    )

    return render_template(
        "reviews.html",
        projects=project_list,
        reviews=review_list
    )


# ABOUT PAGE
@app.route("/about")
def about():
    return render_template("about.html")


# CONTACT PAGE
@app.route("/contact", methods=["GET", "POST"])
def contact():

    if request.method == "POST":

        flash("Message sent successfully.", "success")

        return redirect(url_for("contact"))

    return render_template("contact.html")


# PROFILE PAGE
@app.route("/profile", methods=["GET", "POST"])
def profile():

    login_redirect = require_login()
    if login_redirect:
        return login_redirect

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        dob = request.form.get("dob", "").strip()
        email = request.form.get("email", "").strip()
        contact = request.form.get("contact", "").strip()
        address = request.form.get("address", "").strip()
        gender = request.form.get("gender", "").strip()
        image_path = request.form.get("current_img", "")

        profile_image = request.files.get("profile_image")

        if profile_image and profile_image.filename:

            if allowed_file(profile_image.filename):

                filename = secure_filename(profile_image.filename)
                saved_filename = f"profile_{current_user_id()}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
                saved_path = os.path.join(PROFILE_FOLDER, saved_filename)
                profile_image.save(saved_path)
                image_path = f"static/uploads/profiles/{saved_filename}"

            else:

                flash("Upload JPG, JPEG or PNG image only.", "error")

                return redirect(url_for("profile"))

        db_execute(
            """
            UPDATE users
            SET name=?, dob=?, email=?, contact=?, address=?, gender=?, img=?
            WHERE id=?
            """,
            (name, dob, email, contact, address, gender, image_path, current_user_id())
        )

        session["user"] = name or session.get("user")

        flash("Profile updated successfully.", "success")

        return redirect(url_for("profile"))

    user = db_query("SELECT * FROM users WHERE id=?", (current_user_id(),), one=True)

    return render_template("profile.html", user=user)


# PREDICTION HISTORY PAGE
@app.route("/history")
def history():

    login_redirect = require_login()
    if login_redirect:
        return login_redirect

    predictions = get_user_predictions()

    return render_template("history.html", predictions=predictions)


# ANALYTICS PAGE
@app.route("/analytics")
def analytics():

    login_redirect = require_login()
    if login_redirect:
        return login_redirect

    predictions = get_user_predictions()

    total_predictions = len(predictions)
    average_confidence = 0

    if predictions:

        average_confidence = round(
            sum(row["confidence"] or 0 for row in predictions) / total_predictions,
            2
        )

    severity_chart, confidence_chart = build_analytics_charts(
        predictions,
        current_user_id()
    )

    return render_template(
        "analytics.html",
        total_predictions=total_predictions,
        average_confidence=average_confidence,
        severity_chart=severity_chart,
        confidence_chart=confidence_chart
    )


@app.route("/depth-analysis")
def depth_analysis():

    login_redirect = require_login()
    if login_redirect:
        return login_redirect

    predictions = get_user_predictions()

    forecasts = [estimate_depth_forecast(item) for item in predictions]
    latest_forecast = forecasts[0] if forecasts else None

    return render_template(
        "depth_analysis.html",
        forecasts=forecasts,
        latest_forecast=latest_forecast
    )


# REGISTER PAGE
@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        name = request.form.get("name", "").strip()
        dob = request.form.get("dob", "").strip()
        email = request.form.get("email", "").strip()
        contact = request.form.get("contact", "").strip()
        address = request.form.get("address", "").strip()
        gender = request.form.get("gender", "").strip()
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        # PASSWORD CHECK
        if password != confirm_password:

            flash("Passwords do not match.", "error")

            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password)

        try:
            db_execute(
                """
                INSERT INTO users(username, password, role, name, dob, email, contact, address, gender, img)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (username, hashed_password, 0, name, dob, email, contact, address, gender, "")
            )

            flash("Registration successful. Please login.", "success")

            return redirect(url_for("login"))

        except sql.IntegrityError:

            flash("Username already exists.", "error")

        except Exception as e:

            flash(f"Database error: {str(e)}", "error")

    return render_template("register.html")


# LOGIN PAGE
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        login_id = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        try:
            user = db_query(
                "SELECT * FROM users WHERE username=? OR email=?",
                (login_id, login_id),
                one=True
            )

            # CHECK PASSWORD
            if user and check_password_hash(user["password"], password):

                session["user_id"] = user["id"]
                session["user"] = user["name"] or user["username"]

                flash("Login successful.", "success")

                return redirect(url_for("home"))

            else:

                flash("Invalid email or password.", "error")

        except Exception as e:

            flash(f"Database error: {str(e)}", "error")

    return render_template("login.html")


# LOGOUT
@app.route("/logout")
def logout():

    session.pop("user", None)
    session.pop("user_id", None)

    flash("Logged out successfully.", "success")

    return redirect(url_for("home"))


# PROTECTED PREDICT ROUTE
@app.route("/predict", methods=["POST"])
def predict():

    # LOGIN REQUIRED
    if not current_user_id():

        return jsonify({
            "error": "Please login first."
        }), 401

    # IMAGE CHECK
    if "image" not in request.files:

        return jsonify({
            "error": "No file uploaded"
        }), 400

    file = request.files["image"]

    # EMPTY FILE
    if file.filename == "":

        return jsonify({
            "error": "No selected file"
        }), 400

    # INVALID FILE TYPE
    if not allowed_file(file.filename):

        return jsonify({
            "error": "Invalid file type. Upload PNG, JPG or JPEG."
        }), 400

    filename = secure_filename(file.filename)
    saved_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
    saved_path = os.path.join(UPLOAD_FOLDER, saved_filename)
    file.save(saved_path)

    predicted_filename = f"predicted_{saved_filename}"
    predicted_path = os.path.join(PREDICTED_FOLDER, predicted_filename)

    is_valid_surface, surface_error = validate_surface_image(saved_path)
    if not is_valid_surface:
        return jsonify({
            "error": surface_error
        }), 400

    try:
        status, confidence, severity, log_details = predict_crack_result(
            saved_path,
            predicted_path
        )

    except Exception as e:

        return jsonify({
            "error": f"Image processing failed: {str(e)}"
        }), 500

    try:
        save_prediction_result(
            saved_filename,
            predicted_filename,
            status,
            confidence,
            severity
        )

    except Exception as e:

        return jsonify({
            "error": f"Prediction saved failed: {str(e)}"
        }), 500

    return jsonify({
        "status": status,
        "confidence": confidence,
        "severity": severity,
        "log_details": log_details
    })


# RUN SERVER
if __name__ == "__main__":

    app.run(debug=True, port=5050)










