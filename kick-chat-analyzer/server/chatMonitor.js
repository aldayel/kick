const db = require('./db');
const { fetchKickChat } = require('./kickApi');
const { analyzeMessage } = require('./gptAnalysis');

const activeMonitors = {};

function startMonitoring(channel) {
  if (activeMonitors[channel]) return;
  let lastTimestamp = 0;
  activeMonitors[channel] = setInterval(async () => {
    const messages = await fetchKickChat(channel);
    for (const msg of messages) {
      if (msg.timestamp > lastTimestamp) {
        lastTimestamp = msg.timestamp;
        const analysis = await analyzeMessage(msg.content);
        db.run(
          `INSERT INTO messages (channel, username, message, timestamp, sentiment, emotion, toxicity, engagement) VALUES (?, ?, ?, ?, ?, ?, ?, ?)` ,
          [channel, msg.sender.username, msg.content, msg.timestamp, analysis.sentiment, analysis.emotion, analysis.toxicity, analysis.engagement]
        );
      }
    }
  }, 5000); // poll every 5 seconds
}

function stopMonitoring(channel) {
  if (activeMonitors[channel]) {
    clearInterval(activeMonitors[channel]);
    delete activeMonitors[channel];
  }
}

module.exports = { startMonitoring, stopMonitoring };