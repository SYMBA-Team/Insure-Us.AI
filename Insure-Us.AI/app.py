import numpy as np

from flask import Flask, request, jsonify, render_template
import pickle
import os
# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("./model_pickel.pkl", "rb"))

#@flask_app.route("/")
#def Home():
    #return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The flower species is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(port=5000,debug=True)
   