from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Project folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model
MODEL_PATH = os.path.join(BASE_DIR, "car_price_model.pkl")

model = joblib.load(MODEL_PATH)
feature_columns = list(model.feature_names_in_)

# Check which model file and columns are loaded
print("MODEL FILE:", MODEL_PATH)
print("MODEL COLUMNS:", feature_columns)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Initialize all model features with zero
        input_data = {col: 0 for col in feature_columns}

        # Numeric features from HTML form
        numeric_features = [
            "symboling",
            "wheelbase",
            "carlength",
            "carwidth",
            "carheight",
            "curbweight",
            "enginesize",
            "boreratio",
            "stroke",
            "compressionratio",
            "horsepower",
            "peakrpm",
            "citympg",
            "highwaympg"
        ]

        # Read values entered by the user
        for feature in numeric_features:
            value = request.form.get(feature)

            if value is not None and value.strip() != "":
                if feature in input_data:
                    input_data[feature] = float(value)

        # Default categorical values
        default_categories = [
            "fueltype_gas",
            "aspiration_turbo",
            "doornumber_two",
            "carbody_sedan",
            "drivewheel_fwd",
            "enginelocation_rear",
            "enginetype_ohc",
            "cylindernumber_four",
            "fuelsystem_mpfi"
        ]

        for col in default_categories:
            if col in input_data:
                input_data[col] = 1

        # Create DataFrame in the exact training column order
        input_df = pd.DataFrame(
            [input_data],
            columns=feature_columns
        )

        # Predict car price
        prediction = model.predict(input_df)[0]

        return render_template(
            "index.html",
            prediction=round(float(prediction), 2),
            error=None
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction=None,
            error=str(e)
        ), 400


if __name__ == "__main__":
    app.run(debug=True)