from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model/final_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    feature_names = ['limit_bal', 'sex', 'education', 'marriage', 'age',
                     'pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6',
                     'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6',
                     'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6']

    input_features = [float(request.form[name]) for name in feature_names]
    input_array = np.array(input_features).reshape(1, -1)

    prediction = model.predict(input_array)[0]

    result = "Will Default" if prediction == 1 else "Will Not Default"
    return render_template("index.html", prediction_text=f"Prediction: {result}")

if __name__ == "__main__":
    app.run(debug=True)
