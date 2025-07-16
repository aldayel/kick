const sqlite3 = require('sqlite3').verbose();
const path = require('path');

const db = new sqlite3.Database(path.join(__dirname, 'chat.db'));

db.serialize(() => {
  db.run(`CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel TEXT,
    username TEXT,
    message TEXT,
    timestamp INTEGER,
    sentiment TEXT,
    emotion TEXT,
    toxicity REAL,
    engagement REAL
  )`);
});

module.exports = db;