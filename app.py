from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
app = Flask(__name__)

data = {
    "text": [
        "Pay fee for internship certificate",
        "Guaranteed job after payment",
        "Free internship with training",
        "Learn AI course from university",
        "Limited offer pay now",
        "Official certified course"
    ],
    "label": [1,1,0,0,1,0]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

model = LogisticRegression()
model.fit(X, df["label"])

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = ""

    if request.method == 'POST':
        message = request.form['message']
        vect = vectorizer.transform([message])
        result = model.predict(vect)[0]

        if result == 1:
            prediction = "⚠️ Scam Detected"
        else:
            prediction = "✅ Legitimate Course"

    return render_template("index.html", prediction=prediction)
    
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
