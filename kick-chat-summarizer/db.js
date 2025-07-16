const sqlite3 = require('sqlite3').verbose();
const path = require('path');

const db = new sqlite3.Database(path.join(__dirname, 'chat.db'));

db.serialize(() => {
  db.run(`CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel TEXT,
    username TEXT,
    message TEXT,
    timestamp INTEGER
  )`);
  db.run(`CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel TEXT,
    summary TEXT,
    created_at INTEGER
  )`);
});

module.exports = db;