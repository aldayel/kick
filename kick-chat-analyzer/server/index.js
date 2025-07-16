require('dotenv').config();
const express = require('express');
const cors = require('cors');
const db = require('./db');
const { startMonitoring, stopMonitoring } = require('./chatMonitor');
const http = require('http');
const WebSocket = require('ws');
const { generateTalkingPoints } = require('./gptAnalysis');

const app = express();
const PORT = process.env.PORT || 5001;

app.use(cors());
app.use(express.json());

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

app.post('/api/monitor/:channel', (req, res) => {
  startMonitoring(req.params.channel);
  res.json({ status: 'monitoring', channel: req.params.channel });
});

app.post('/api/stop/:channel', (req, res) => {
  stopMonitoring(req.params.channel);
  res.json({ status: 'stopped', channel: req.params.channel });
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

server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});