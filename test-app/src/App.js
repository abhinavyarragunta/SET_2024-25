import React, { useState, useEffect } from 'react';
import logo from './logo.svg';
import './App.css';

function App() {
  const [ipAddress, setIpAddress] = useState('');
  const flaskServerUrl = "http://127.0.0.1:8000";

  useEffect(() => {
    fetch(`${flaskServerUrl}/get-ip`)
      .then(response => response.json())
      .then(data => setIpAddress(data.ip))
      .catch(error => console.error("Error fetching IP:", error));
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <div className="video-feed">
          <h2>Live Video Feed</h2>
          <img
          src={`${flaskServerUrl}/stream`}
          alt="Live Video Feed"
          onError={(e) => console.error("Error loading video feed:", e)}
          onLoad={() => console.log("Video feed loaded successfully")}
          crossOrigin="anonymous"
          style={{ width: '80%', border: '2px solid #333' }}
          />
        </div>
      </header>
    </div>
  );
}

export default App;
