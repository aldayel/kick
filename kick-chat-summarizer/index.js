require('dotenv').config();
const express = require('express');
const cors = require('cors');
const db = require('./db');
const { connectToKickChat } = require('./kickWs');
const { summarizeMessages } = require('./gpt');
const path = require('path');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 5001;

app.use(cors());
app.use(express.json());

// In-memory state for active channel and summary
let activeChannel = null;
let summary = '';
let lastSummaryTime = 0;

// Start chat listener for a channel
function startChannel(channel) {
  if (activeChannel === channel) return;
  activeChannel = channel;
  connectToKickChat(channel, (msg) => {
    db.run(
      `INSERT INTO messages (channel, username, message, timestamp) VALUES (?, ?, ?, ?)`,
      [channel, msg.username, msg.message, msg.timestamp]
    );
  });
}

// Summarize every 30 seconds
setInterval(async () => {
  if (!activeChannel) return;
  db.all(
    'SELECT username, message FROM messages WHERE channel = ? AND timestamp > ?',
    [activeChannel, Date.now() - 5 * 60 * 1000], // last 5 min
    async (err, rows) => {
      if (err || rows.length === 0) return;
      try {
        const sum = await summarizeMessages(rows);
        summary = sum;
        lastSummaryTime = Date.now();
        db.run(
          `INSERT INTO summaries (channel, summary, created_at) VALUES (?, ?, ?)`,
          [activeChannel, sum, Date.now()]
        );
      } catch (e) {
        // Ignore summarization errors
      }
    }
  );
}, 30000);

app.post('/api/channel', (req, res) => {
  const { channel } = req.body;
  if (!channel) return res.status(400).json({ error: 'Missing channel' });
  startChannel(channel);
  res.json({ status: 'started', channel });
});

app.get('/api/messages/:channel', (req, res) => {
  db.all(
    'SELECT * FROM messages WHERE channel = ? ORDER BY timestamp DESC LIMIT 100',
    [req.params.channel],
    (err, rows) => {
      if (err) return res.status(500).json({ error: err.message });
      res.json(rows);
    }
  );
});

app.get('/api/summary/:channel', (req, res) => {
  db.get(
    'SELECT summary, created_at FROM summaries WHERE channel = ? ORDER BY created_at DESC LIMIT 1',
    [req.params.channel],
    (err, row) => {
      if (err || !row) return res.json({ summary: '', updated: 0 });
      res.json({ summary: row.summary, updated: row.created_at });
    }
  );
});

// Webhook endpoint for Kick.com
app.post('/kick-webhook', (req, res) => {
  const eventType = req.body.type || 'unknown';
  const eventData = JSON.stringify(req.body);
  const receivedAt = Date.now();
  db.run(
    `INSERT INTO webhook_events (event_type, event_data, received_at) VALUES (?, ?, ?)` ,
    [eventType, eventData, receivedAt]
  );
  console.log('Received webhook from Kick.com:', req.body);
  res.status(200).send('OK');
});

// Fetch latest webhook events
app.get('/api/webhook-events', (req, res) => {
  db.all(
    'SELECT * FROM webhook_events ORDER BY received_at DESC LIMIT 100',
    [],
    (err, rows) => {
      if (err) return res.status(500).json({ error: err.message });
      res.json(rows);
    }
  );
});

// OAuth callback endpoint for Kick.com
app.get('/oauth/callback', async (req, res) => {
  const code = req.query.code;
  if (!code) {
    return res.send('<h2>Authorization failed: No code provided.</h2>');
  }
  try {
    const tokenRes = await axios.post('https://kick.com/oauth2/token', null, {
      params: {
        grant_type: 'authorization_code',
        code,
        redirect_uri: 'https://chatly-e0ut.onrender.com/oauth/callback',
        client_id: process.env.KICK_CLIENT_ID || '01K09KDXDAQAWJF9V1VAD0TEFV',
        client_secret: process.env.KICK_CLIENT_SECRET || '45d847c20951ae22a8a9644639de2c03795a31a050cbfe8114dee41d365fed52'
      }
    });
    const { access_token, refresh_token, expires_in, user_id } = tokenRes.data;
    db.run(
      `INSERT INTO oauth_tokens (user_id, access_token, refresh_token, expires_at, created_at) VALUES (?, ?, ?, ?, ?)`,
      [user_id || '', access_token, refresh_token, Date.now() + expires_in * 1000, Date.now()]
    );
    res.send('<h2>Authorization successful! You can close this window.</h2>');
  } catch (e) {
    console.error('OAuth token exchange failed:', e.response?.data || e.message);
    res.send('<h2>Authorization failed: Could not exchange code for token.</h2>');
  }
});

// Fetch chatroom info for a given channel using the official Kick API
app.get('/api/chatroom/:channel', async (req, res) => {
  const channel = req.params.channel;
  try {
    const response = await axios.get(`https://kick.com/api/v2/channels/${channel}`);
    res.json(response.data);
  } catch (e) {
    res.status(500).json({ error: e.response?.data || e.message });
  }
});

// Serve React frontend static files
app.use(express.static(path.join(__dirname, 'kick-chat-frontend', 'build')));

// Only serve index.html for non-API, non-webhook GET requests
app.get(/^\/(?!api|kick-webhook).*/, (req, res) => {
  res.sendFile(path.join(__dirname, 'kick-chat-frontend', 'build', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});