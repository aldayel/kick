import { useState, useEffect, useRef } from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, BarChart, Bar, XAxis, YAxis } from 'recharts';
import { FaRobot, FaSmile, FaFrown, FaMeh, FaComments, FaChartBar } from 'react-icons/fa';
import logo from './assets/react.svg';
import './App.css';

const SENTIMENT_COLORS = {
  positive: '#2ecc40',
  negative: '#ff4136',
  neutral: '#888',
};

function App() {
  const [channel, setChannel] = useState('');
  const [messages, setMessages] = useState([]);
  const [stats, setStats] = useState(null);
  const [talkingPoints, setTalkingPoints] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const chatBoxRef = useRef(null);

  // Auto-scroll chat to bottom
  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    if (!channel) return;
    setLoading(true);
    setError('');
    const fetchMessages = async () => {
      try {
        const res = await fetch(`/api/messages/${channel}`);
        setMessages(await res.json());
      } catch (e) {
        setError('Failed to fetch messages.');
      }
    };
    const fetchStats = async () => {
      try {
        const res = await fetch(`/api/stats/${channel}`);
        setStats(await res.json());
      } catch (e) {
        setError('Failed to fetch stats.');
      }
    };
    const fetchTalkingPoints = async () => {
      try {
        const res = await fetch(`/api/talking-points/${channel}`);
        setTalkingPoints(await res.json());
      } catch (e) {
        setError('Failed to fetch talking points.');
      }
    };
    Promise.all([fetchMessages(), fetchStats(), fetchTalkingPoints()]).finally(() => setLoading(false));
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
  }, [channel]);

  // Prepare data for charts
  const sentimentData = stats ? [
    { name: 'Positive', value: stats.positive, color: SENTIMENT_COLORS.positive },
    { name: 'Negative', value: stats.negative, color: SENTIMENT_COLORS.negative },
    { name: 'Neutral', value: stats.neutral, color: SENTIMENT_COLORS.neutral },
  ] : [];
  const toxicityData = stats ? [
    { name: 'Toxicity', value: stats.avgToxicity },
    { name: 'Non-toxic', value: 1 - stats.avgToxicity },
  ] : [];

  return (
    <div className="app-bg">
      <header className="header">
        <img src={logo} alt="Logo" className="header-logo" />
        <span className="header-title">Kick.com Live Chat Analyzer</span>
      </header>
      <div className="container">
        <div className="input-group">
          <input
            type="text"
            placeholder="Enter Kick.com stream channel name"
            value={channel}
            onChange={e => setChannel(e.target.value.replace('https://kick.com/', ''))}
            disabled={loading}
          />
        </div>
        {error && <div className="error-alert">{error}</div>}
        <div className="main-content">
          <div className="chat-section">
            <h2><FaComments /> Live Chat</h2>
            <div className="chat-box" ref={chatBoxRef}>
              {loading && <div className="loading">Loading chat...</div>}
              {!loading && messages.length === 0 && <div className="empty">No messages yet. Waiting for Python service to ingest chat...</div>}
              {messages.map(msg => (
                <div key={msg.id} className={`chat-msg ${msg.sentiment}`}>
                  <div className="chat-msg-header">
                    <span className="user-avatar">{msg.username ? msg.username[0].toUpperCase() : '?'}</span>
                    <span className="user">{msg.username}:</span>
                  </div>
                  <span className="msg-text">{msg.message}</span>
                  <span className="meta">
                    {msg.sentiment === 'positive' && <FaSmile color={SENTIMENT_COLORS.positive} title="Positive" />} 
                    {msg.sentiment === 'negative' && <FaFrown color={SENTIMENT_COLORS.negative} title="Negative" />} 
                    {msg.sentiment === 'neutral' && <FaMeh color={SENTIMENT_COLORS.neutral} title="Neutral" />} 
                    {msg.sentiment}, {msg.emotion}, tox: {(msg.toxicity * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
          <div className="stats-section">
            <h2><FaChartBar /> Stats</h2>
            {stats ? (
              <>
                <div className="chart-row">
                  <div className="chart-card">
                    <h4>Sentiment</h4>
                    <ResponsiveContainer width="100%" height={180}>
                      <PieChart>
                        <Pie data={sentimentData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={60} label>
                          {sentimentData.map((entry, idx) => (
                            <Cell key={`cell-${idx}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="chart-card">
                    <h4>Toxicity</h4>
                    <ResponsiveContainer width="100%" height={180}>
                      <BarChart data={toxicityData} layout="vertical">
                        <XAxis type="number" domain={[0, 1]} hide />
                        <YAxis type="category" dataKey="name" hide />
                        <Bar dataKey="value" fill="#ff4136" radius={8} />
                        <Tooltip />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                <ul className="stats-list">
                  <li><b>Total messages:</b> {stats.total}</li>
                  <li><b>Positive:</b> {(stats.positive * 100).toFixed(1)}%</li>
                  <li><b>Negative:</b> {(stats.negative * 100).toFixed(1)}%</li>
                  <li><b>Neutral:</b> {(stats.neutral * 100).toFixed(1)}%</li>
                  <li><b>Avg Toxicity:</b> {(stats.avgToxicity * 100).toFixed(1)}%</li>
                </ul>
              </>
            ) : <p>No stats yet.</p>}
          </div>
          <div className="talking-points-section">
            <h2><FaRobot /> AI Talking Points</h2>
            <div className="tp-list">
              {talkingPoints.length === 0 && <div className="empty">No talking points yet.</div>}
              {talkingPoints.map((tp, i) => (
                <div className="tp-card" key={i}>
                  <span className="tp-emoji">💡</span> {tp}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
