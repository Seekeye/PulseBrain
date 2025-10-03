from flask import Flask, jsonify
import os
from datetime import datetime

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "CryptoPulse Pro Bot"
    })

@app.route('/')
def home():
    return jsonify({"message": "CryptoPulse Pro Bot is running"})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
