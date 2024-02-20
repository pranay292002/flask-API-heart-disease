from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the model and scaler
try:
    with open('heart_disease_model.pkl', 'rb') as model_file:
        clf = pickle.load(model_file)
    with open('heart_disease_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    print("Error: Model files not found. Make sure you have the necessary model files in the correct location.")
    clf = None
    scaler = None

@app.route('/predict', methods=['POST'])
def predict():
    if clf is None or scaler is None:
        return jsonify("Error: Model not loaded. Please check server logs for details.")

    data = request.get_json(force=True)
    age = data['age']
    sex = data['sex']
    chest_pain_type = data['chest_pain_type']
    resting_blood_pressure = data['resting_blood_pressure']
    cholesterol = data['cholesterol']
    fasting_blood_sugar = data['fasting_blood_sugar']
    resting_ecg = data['resting_ecg']
    max_heart_rate = data['max_heart_rate']
    exercise_induced_angina = data['exercise_induced_angina']
    st_depression = data['st_depression']
    st_slope = data['st_slope']
    num_major_vessels = data['num_major_vessels']
    thal = data['thal']

    user_input = np.array([[age, sex, chest_pain_type, resting_blood_pressure, cholesterol, fasting_blood_sugar, resting_ecg, max_heart_rate, exercise_induced_angina, st_depression, st_slope, num_major_vessels, thal]])
    user_input_scaled = scaler.transform(user_input)
    
    try:
        prediction = clf.predict(user_input_scaled)

        if prediction == 0:
            result = "Good News! You don't have Heart Disease."
        else:
            result = "Sorry to say, You may have Heart Disease."

        return jsonify(result)
    except Exception as e:
        return jsonify("Error: {}".format(str(e)))

if __name__ == '__main__':
    app.run(port=5000, debug=True)
