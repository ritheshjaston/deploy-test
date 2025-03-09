from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"message": "Emotion Detection API is running!"})

# Vercel-specific handler
def handler(event, context):
    return app(event, context)
