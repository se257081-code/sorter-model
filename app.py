from flask import Flask, request
import joblib, pandas as pd

app = Flask(__name__)
clf = joblib.load("clf.pkl")
reg = joblib.load("reg.pkl")

@app.route("/")
def home():
    return "Smart Sorter model is running"

@app.route("/predict")
def predict():
    w = float(request.args.get("weight", 0))
    X = pd.DataFrame({"weight": [w]})
    action = clf.predict(X)[0]
    amount = int(round(reg.predict(X)[0]))
    if action == "Add":
        return f"Add: {amount} gram"
    if action == "Remove":
        return f"Remove: {amount} grams"
    return "OK"