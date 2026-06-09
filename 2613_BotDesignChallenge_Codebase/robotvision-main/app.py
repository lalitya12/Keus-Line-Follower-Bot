from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import logging
import time
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'kriti2026_secret'
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=1e8)

# Disable default werkzeug logging to keep console clean
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

LOG_FILE = "dashboard_log.txt"

def write_to_log(text):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}\n")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")
    
@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")

@socketio.on('video_frame')
def handle_video_frame(data):
    # data expects {'image': 'base64 string...'}
    emit('update_frame', data, broadcast=True)

@socketio.on('recognition_result')
def handle_recognition(data):
    # data expects {'text': '...', 'image': 'base64 string...'}
    print(f"Recognition received:\n{data.get('text', '')}")
    if 'text' in data:
        lines = data['text'].split('\n')
        for line in lines:
            if line.strip():
                write_to_log(line.strip())
        
    emit('new_recognition', data, broadcast=True)

if __name__ == '__main__':
    print("Starting Kriti 2026 Dashboard on http://0.0.0.0:8080")
    if not os.path.exists(LOG_FILE):
        open(LOG_FILE, 'w').close()
    socketio.run(app, host='0.0.0.0', port=8080, debug=False)
