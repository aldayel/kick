import os
import requests

def send_to_backend(channel, username, message, timestamp):
    backend_url = os.environ.get('BACKEND_URL', 'http://localhost:5001/api/ingest-message')
    ingest_token = os.environ.get('INGEST_TOKEN', 'changeme')
    data = {
        'channel': channel,
        'username': username,
        'message': message,
        'timestamp': timestamp
    }
    headers = {'x-ingest-token': ingest_token}
    try:
        resp = requests.post(backend_url, json=data, headers=headers, timeout=2)
        resp.raise_for_status()
        print(f"[Ingest] Sent message from {username}")
    except Exception as e:
        print(f"[Ingest] Failed to send message: {e}")