require('dotenv').config();
const express = require('express');
const cors = require('cors');
const db = require('./db');
const { connectToKickChat } = require('./kickWs');
const { summarizeMessages } = require('./gpt');

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
  console.log('Received webhook from Kick.com:', req.body);
  res.status(200).send('OK');
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});