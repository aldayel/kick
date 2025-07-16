from send_to_backend import send_to_backend

def on_new_message(msg):
    # msg should have 'username', 'message', 'timestamp', and 'channel' fields
    send_to_backend(
        channel=msg.get('channel', 'default'),
        username=msg['username'],
        message=msg['message'],
        timestamp=msg['timestamp']
    )

# ... in your chat reading loop, call on_new_message(msg) for each new message