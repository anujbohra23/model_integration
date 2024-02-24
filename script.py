from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)


# Prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 12)
    loaded_model = pickle.load(open("model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


# Route for handling POST requests and making predictions
@app.route("/result", methods=["POST"])
def result():
    if request.method == "POST":
        # Get data from the form
        to_predict_list = list(request.form.values())
        # Convert data to integers
        to_predict_list = list(map(int, to_predict_list))
        # Make prediction
        result = ValuePredictor(to_predict_list)
        # Interpret prediction
        if result == 1:
            prediction = "Income more than 50K"
        else:
            prediction = "Income less than 50K"
        # Render result.html with prediction
        return render_template("result.html", prediction=prediction)


# Route for rendering the form
@app.route("/")
def form():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
