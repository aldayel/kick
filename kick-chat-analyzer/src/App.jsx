import { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [channel, setChannel] = useState('');
  const [monitoring, setMonitoring] = useState(false);
  const [messages, setMessages] = useState([]);
  const [stats, setStats] = useState(null);
  const [talkingPoints, setTalkingPoints] = useState([]);
  const ws = useRef(null);

  useEffect(() => {
    if (!monitoring || !channel) return;
    const fetchMessages = async () => {
      const res = await fetch(`/api/messages/${channel}`);
      setMessages(await res.json());
    };
    const fetchStats = async () => {
      const res = await fetch(`/api/stats/${channel}`);
      setStats(await res.json());
    };
    const fetchTalkingPoints = async () => {
      const res = await fetch(`/api/talking-points/${channel}`);
      setTalkingPoints(await res.json());
    };
    fetchMessages();
    fetchStats();
    fetchTalkingPoints();
    const interval = setInterval(() => {
      fetchMessages();
      fetchStats();
    }, 5000);
    const tpInterval = setInterval(() => {
      fetchTalkingPoints();
    }, 15000);
    return () => {
      clearInterval(interval);
      clearInterval(tpInterval);
    };
  }, [monitoring, channel]);

  const handleStart = async () => {
    const ch = channel.replace('https://kick.com/', '');
    setChannel(ch);
    await fetch(`/api/monitor/${ch}`, { method: 'POST' });
    setMonitoring(true);
  };
  const handleStop = async () => {
    await fetch(`/api/stop/${channel}`, { method: 'POST' });
    setMonitoring(false);
  };

  return (
    <div className="container">
      <h1>Kick.com Live Chat Analyzer</h1>
      <div className="input-group">
        <input
          type="text"
          placeholder="Enter Kick.com stream URL or channel name"
          value={channel}
          onChange={e => setChannel(e.target.value)}
          disabled={monitoring}
        />
        {!monitoring ? (
          <button onClick={handleStart} disabled={!channel}>Start Monitoring</button>
        ) : (
          <button onClick={handleStop}>Stop</button>
        )}
      </div>
      <div className="main-content">
        <div className="chat-section">
          <h2>Live Chat</h2>
          <div className="chat-box">
            {messages.map(msg => (
              <div key={msg.id} className={`chat-msg ${msg.sentiment}`}>
                <span className="user">{msg.username}:</span> {msg.message}
                <span className="meta">[{msg.sentiment}, {msg.emotion}, tox: {msg.toxicity}]</span>
              </div>
            ))}
          </div>
        </div>
        <div className="stats-section">
          <h2>Stats</h2>
          {stats ? (
            <ul>
              <li>Total messages: {stats.total}</li>
              <li>Positive: {(stats.positive * 100).toFixed(1)}%</li>
              <li>Negative: {(stats.negative * 100).toFixed(1)}%</li>
              <li>Neutral: {(stats.neutral * 100).toFixed(1)}%</li>
              <li>Avg Toxicity: {(stats.avgToxicity * 100).toFixed(1)}%</li>
            </ul>
          ) : <p>No stats yet.</p>}
        </div>
        <div className="talking-points-section">
          <h2>AI Talking Points</h2>
          <ul>
            {talkingPoints.map((tp, i) => <li key={i}>{tp}</li>)}
          </ul>
        </div>
      </div>
    </div>
  );
}

export default App;
