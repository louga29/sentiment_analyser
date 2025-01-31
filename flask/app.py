from flask import Flask, render_template, request, redirect, url_for, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

app = Flask(__name__)


pipeline_path = "text_classification_pipeline.pkl"
pipeline = joblib.load(pipeline_path)

uri = "mongodb+srv://arthurgautier29480:lapin@cluster0.wuq03.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
mydb = client.get_database("sentiment_analyser")
collection = mydb["reviews"]

@app.route("/", methods=["GET"])
def index():
    print("Page d'accueil chargée")
    return render_template("index.html")

@app.route("/result", methods=["GET", "POST"])
def result():
    if request.method == "POST":
        phrase = request.form.get("label_phrase")
        
        if phrase:
            prediction = pipeline.predict([phrase])[0]
            print(phrase)
            return render_template(
                "result.html", 
                phrase_on_page=phrase, 
                prediction_on_page=prediction, 
            )
        else:
            return render_template(
                "result.html", 
                phrase_on_page="Aucune phrase entrée", 
                prediction_on_page="Erreur", 
            )
        
    elif request.method == "GET":
        phrase = request.args.get("phrase")
        
        if phrase:
            prediction = pipeline.predict([phrase])[0]

            return jsonify({"text": phrase, "prediction": prediction})
        else:
            return jsonify({"error": "Aucune phrase fournie"}), 400
    
    else:
        return redirect(url_for("index"))

@app.route("/save_feedback", methods=["POST"])
def save_feedback():
    # Get the user inputs and feedback
    user_input = request.form.get("phrase_on_page")
    prediction = request.form.get("prediction_on_page")
    feedback = request.form.get("feedback")

    print(f"User input: {user_input}")
    print(f"Prediction: {prediction} ")
    print(f"Feedback: {feedback}")

    # Store data in MongoDB
    if user_input and prediction and feedback:
        feedback_data = {
            "user_input": user_input,
            "prediction": prediction,
            "feedback": feedback
        }
        collection.insert_one(feedback_data)
        print("Feedback saved in MongoDB")  # Debugging
        return redirect(url_for('index'))  # Redirect back to the homepage after saving feedback
    else:
        return render_template("index.html", prediction="Please make a prediction first and provide feedback.", user_input=user_input)


if __name__ == '__main__':
    app.run(debug=True)
