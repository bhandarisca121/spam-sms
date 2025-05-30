from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

model_path = os.path.join('model', 'spam_classifier_model.pkl')
model = joblib.load(model_path)
# ✅ Direct paths to the .pkl files (not a directory)
model_path = "model/spam_classifier_model.pkl"
vectorizer_path = "model/tfidf_vectorizer.pkl"

# Load the actual model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        message = request.form["message"]
        message_transformed = vectorizer.transform([message])
        pred = model.predict(message_transformed)[0]
        prediction = "Spam" if pred == 1 else "Not Spam"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
