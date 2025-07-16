require('dotenv').config();
const express = require('express');
const cors = require('cors');
const db = require('./db');
const { startMonitoring, stopMonitoring } = require('./chatMonitor');
const http = require('http');
const WebSocket = require('ws');
const { generateTalkingPoints } = require('./gptAnalysis');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 5001;

app.use(cors());
app.use(express.json());

app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
  next();
});

const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Store connected clients
const clients = new Set();
wss.on('connection', (ws) => {
  clients.add(ws);
  ws.on('close', () => clients.delete(ws));
});

// Broadcast helper
function broadcast(data) {
  for (const client of clients) {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify(data));
    }
  }
}

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok' });
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

app.get('/api/stats/:channel', (req, res) => {
  db.all(
    'SELECT sentiment, toxicity FROM messages WHERE channel = ?',
    [req.params.channel],
    (err, rows) => {
      if (err) return res.status(500).json({ error: err.message });
      let total = rows.length;
      let pos = 0, neg = 0, neu = 0, tox = 0;
      for (const row of rows) {
        if (row.sentiment === 'positive') pos++;
        else if (row.sentiment === 'negative') neg++;
        else neu++;
        tox += row.toxicity || 0;
      }
      res.json({
        total,
        positive: total ? (pos / total) : 0,
        negative: total ? (neg / total) : 0,
        neutral: total ? (neu / total) : 0,
        avgToxicity: total ? (tox / total) : 0,
      });
    }
  );
});

app.get('/api/talking-points/:channel', async (req, res) => {
  db.all(
    'SELECT message FROM messages WHERE channel = ? ORDER BY timestamp DESC LIMIT 30',
    [req.params.channel],
    async (err, rows) => {
      if (err) return res.status(500).json({ error: err.message });
      const talkingPoints = await generateTalkingPoints(rows);
      res.json(talkingPoints);
    }
  );
});

// Secure ingest endpoint for Python chat reader
app.post('/api/ingest-message', async (req, res) => {
  // Optional: Use a shared secret for authentication
  const AUTH_TOKEN = process.env.INGEST_TOKEN || 'changeme';
  if (req.headers['x-ingest-token'] !== AUTH_TOKEN) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  const { channel, username, message, timestamp } = req.body;
  if (!channel || !username || !message || !timestamp) {
    return res.status(400).json({ error: 'Missing required fields' });
  }
  try {
    const { analyzeMessage } = require('./gptAnalysis');
    const analysis = await analyzeMessage(message);
    db.run(
      `INSERT INTO messages (channel, username, message, timestamp, sentiment, emotion, toxicity, engagement) VALUES (?, ?, ?, ?, ?, ?, ?, ?)` ,
      [channel, username, message, timestamp, analysis.sentiment, analysis.emotion, analysis.toxicity, analysis.engagement]
    );
    // Optionally broadcast to WebSocket clients
    if (typeof broadcast === 'function') {
      broadcast({ type: 'new_message', data: { channel, username, message, timestamp, ...analysis } });
    }
    res.json({ status: 'ok' });
  } catch (e) {
    console.error('Ingest error:', e);
    res.status(500).json({ error: 'Failed to analyze or store message.' });
  }
});

if (process.env.NODE_ENV === 'production') {
  const clientBuildPath = path.join(__dirname, '../dist');
  app.use(express.static(clientBuildPath));
  app.get('*', (req, res) => {
    res.sendFile(path.join(clientBuildPath, 'index.html'));
  });
}

server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});