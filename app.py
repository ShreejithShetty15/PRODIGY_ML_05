from flask import (
    Flask,
    render_template,
    request,
    send_from_directory
)

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

from tensorflow.keras.preprocessing import image

import numpy as np
import os

# =========================
# APP CONFIG
# =========================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =========================
# LOAD AI MODEL
# =========================

model = MobileNetV2(weights="imagenet")

# =========================
# FOOD DATABASE
# =========================

food_info = {

    "pizza": {
        "calories": 285,
        "protein": "12g",
        "carbs": "36g",
        "fat": "10g"
    },

    "hamburger": {
        "calories": 295,
        "protein": "17g",
        "carbs": "30g",
        "fat": "14g"
    },

    "cheeseburger": {
        "calories": 320,
        "protein": "18g",
        "carbs": "31g",
        "fat": "16g"
    },

    "hotdog": {
        "calories": 250,
        "protein": "10g",
        "carbs": "22g",
        "fat": "15g"
    },

    "ice_cream": {
        "calories": 207,
        "protein": "4g",
        "carbs": "24g",
        "fat": "11g"
    }

}

# =========================
# IMAGE PREVIEW ROUTE
# =========================

@app.route('/uploads/<filename>')
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )

# =========================
# MAIN PAGE
# =========================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        file = request.files["image"]

        if file.filename == "":
            return render_template(
                "index.html",
                result=None
            )

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            file.filename
        )

        file.save(filepath)

        # =====================
        # PREPROCESS IMAGE
        # =====================

        img = image.load_img(
            filepath,
            target_size=(224, 224)
        )

        x = image.img_to_array(img)

        x = np.expand_dims(
            x,
            axis=0
        )

        x = preprocess_input(x)

        # =====================
        # AI PREDICTION
        # =====================

        preds = model.predict(x)

        prediction = decode_predictions(
            preds,
            top=1
        )[0][0]

        label = prediction[1]

        confidence = round(
            prediction[2] * 100,
            2
        )

        # =====================
        # NUTRITION LOOKUP
        # =====================

        nutrition = food_info.get(
            label,
            {
                "calories": "N/A",
                "protein": "-",
                "carbs": "-",
                "fat": "-"
            }
        )

        # =====================
        # RESULT OBJECT
        # =====================

        result = {

            "label": label,

            "confidence": confidence,

            "nutrition": nutrition,

            "image": file.filename

        }

    return render_template(
        "index.html",
        result=result
    )

# =========================
# RUN APP
# =========================

if __name__ == "__main__":

    app.run(
        debug=True
    )