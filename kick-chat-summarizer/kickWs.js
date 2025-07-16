const WebSocket = require('ws');

function connectToKickChat(channel, onMessage) {
  const ws = new WebSocket('wss://chat.kick.com');

  ws.on('open', () => {
    ws.send(JSON.stringify({
      event: 'join',
      data: { room: `channel:${channel}` }
    }));
  });

  ws.on('message', (data) => {
    try {
      const msg = JSON.parse(data);
      if (msg.event === 'message' && msg.data && msg.data.content) {
        onMessage({
          username: msg.data.sender && msg.data.sender.username,
          message: msg.data.content,
          timestamp: Date.now(),
          channel
        });
      }
    } catch (e) {
      // Ignore parse errors
    }
  });

  ws.on('error', (err) => {
    console.error('Kick WS error:', err);
  });

  ws.on('close', () => {
    console.log('Kick WS closed, reconnecting in 5s...');
    setTimeout(() => connectToKickChat(channel, onMessage), 5000);
  });

  return ws;
}

module.exports = { connectToKickChat };