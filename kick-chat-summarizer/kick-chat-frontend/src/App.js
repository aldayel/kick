import React, { useEffect, useState } from 'react';
import './App.css';

function formatDate(ts) {
  const d = new Date(ts);
  return d.toLocaleString();
}

function App() {
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchEvents = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/webhook-events');
      const data = await res.json();
      setEvents(data);
    } catch (e) {
      setEvents([]);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchEvents();
    const interval = setInterval(fetchEvents, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="App">
      <h1>Kick.com Webhook Event Dashboard</h1>
      {loading ? <p>Loading...</p> : null}
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th>ID</th>
            <th>Type</th>
            <th>Received At</th>
            <th>Data</th>
          </tr>
        </thead>
        <tbody>
          {events.map(ev => (
            <tr key={ev.id}>
              <td>{ev.id}</td>
              <td>{ev.event_type}</td>
              <td>{formatDate(ev.received_at)}</td>
              <td>
                <pre style={{ maxWidth: 400, whiteSpace: 'pre-wrap', wordBreak: 'break-all', fontSize: 12 }}>
                  {ev.event_data}
                </pre>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default App;
