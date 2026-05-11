from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("diabetes_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    pregnancies = float(request.form["Pregnancies"])
    glucose = float(request.form["Glucose"])
    blood_pressure = float(request.form["BloodPressure"])
    skin_thickness = float(request.form["SkinThickness"])
    insulin = float(request.form["Insulin"])
    bmi = float(request.form["BMI"])
    dpf = float(request.form["DiabetesPedigreeFunction"])
    age = float(request.form["Age"])

    features = np.array([[pregnancies, glucose, blood_pressure,
                          skin_thickness, insulin, bmi, dpf, age]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "Diabetic"
    else:
        result = "Non-Diabetic"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)