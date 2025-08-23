import os, sys, time, json
from pathlib import Path
from openai import OpenAI


# --- make sure we can import utils/* even when running python app/app.py ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

# your original inference function (unchanged)
from utils.inference import predict_with_cam

# DB helpers (new)
from utils.db import init_db, log_prediction, get_history

# LLM support (optional, only used if key exists)
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Lazy import so missing packages don’t crash the app
_openai = None
_genai = None
if OPENAI_API_KEY:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        _openai = openai
    except Exception:
        _openai = None
if (not _openai) and GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        _genai = genai
    except Exception:
        _genai = None

# -----------------------------------------------------------------------------
# Flask setup
# -----------------------------------------------------------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
OVERLAY_DIR = BASE_DIR / "static" / "overlays"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

# Init DB table if not present
init_db()


def generate_explanation(pred_label, probs_dict):
    """Generate natural language explanation using OpenAI (preferred) or Gemini (fallback)."""
    explanation = None

    # ---- Try OpenAI first ----
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            client = OpenAI(api_key=openai_key)

            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a medical assistant. Explain chest X-ray predictions in clear, factual, non-diagnostic language. Keep it simple and professional."},
                    {"role": "user", "content": f"The model predicted: {pred_label}. Probabilities: {probs_dict}"}
                ],
                temperature=0.3,
            )

            explanation = response.choices[0].message.content.strip()
            return explanation

        except Exception as e:
            return f"Explanation unavailable (OpenAI API error): {str(e)}"

    # ---- Fallback to Gemini ----
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            genai.configure(api_key=gemini_key)

            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(
                f"Explain this chest X-ray prediction in simple, factual terms (not diagnostic). "
                f"Prediction: {pred_label}. Probabilities: {probs_dict}"
            )

            explanation = response.text
            return explanation

        except Exception as e:
            return f"Explanation unavailable (Gemini API error): {str(e)}"

    return "Explanation unavailable: No API key configured."

@app.route("/", methods=["GET", "POST"])
def index():
    context = {"prediction": None, "probs": None, "overlay_url": None, "explanation": None}
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            context["explanation"] = "Please choose an image file (JPG or PNG)."
            return render_template("index.html", **context)

        fname = f"{int(time.time())}_{secure_filename(file.filename)}"
        save_path = UPLOAD_DIR / fname
        file.save(str(save_path))

        # ---- inference + Grad-CAM (uses your original code) ----
        top_label, probs_dict, overlay_np = predict_with_cam(str(save_path))

        # Save overlay image beside static/overlays
        overlay_name = f"{Path(fname).stem}_overlay.jpg"
        overlay_path = OVERLAY_DIR / overlay_name

        # cv2 is used inside predict_with_cam; just save here safely
        try:
            import cv2
            cv2.imwrite(str(overlay_path), overlay_np)
        except Exception:
            # if OpenCV missing for save (unlikely), skip saving overlay
            overlay_path = None

        # Build context for template
        context["prediction"] = top_label
        context["probs"] = probs_dict
        context["overlay_url"] = (
            url_for("static", filename=f"overlays/{overlay_name}") if overlay_path and overlay_path.exists() else None
        )
        context["explanation"] = generate_explanation(top_label, probs_dict)

        # Log in DB (simple JSON string for probabilities)
        try:
            log_prediction(
                label=top_label,
                probs_json=json.dumps(probs_dict),
                image_name=fname,
                overlay_name=overlay_name if overlay_path else None,
            )
        except Exception:
            pass

    return render_template("index.html", **context)


@app.route("/history", methods=["GET"])
def history():
    rows = get_history()
    # rows: id, label, probs, image_name, overlay_name, timestamp
    return render_template("history.html", rows=rows)


if __name__ == "__main__":
    # Run the flask app as before
    app.run(host="127.0.0.1", port=5000, debug=True)
