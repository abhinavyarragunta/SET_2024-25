import React, { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';
import './App.css';
import DirectionButtons from './DirectionButtons';

function App() {
  const socket = useRef(null);
  const audioContextRef = useRef(null);
  const [ipAddress, setIpAddress] = useState('');
  const [isVideoOn, setIsVideoOn] = useState(false);
  const [isAudioOn, setIsAudioOn] = useState(false);
  const flaskServerUrl = "http://127.0.0.1:8000";

  // Fetch the IP address from the Flask server when the component mounts
  useEffect(() => {
    fetch(`${flaskServerUrl}/get-ip`)
      .then(response => response.json())
      .then(data => setIpAddress(data.ip))
      .catch(error => console.error("Error fetching IP:", error));

    // Initialize WebSocket connection for audio data
    socket.current = io(flaskServerUrl);

    // Initialize AudioContext
    audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();

    // Listen for audio data from server
    socket.current.on('audio_data', (data) => {
      console.log("Received audio data:", data); // Log to check if audio data is received
      const floatData = Float32Array.from(data);
      const audioBuffer = audioContextRef.current.createBuffer(1, floatData.length, 44100);
      audioBuffer.getChannelData(0).set(floatData);

      // Play the audio data
      const source = audioContextRef.current.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContextRef.current.destination);
      source.start();
    });

    // Cleanup WebSocket and AudioContext when component unmounts
    return () => {
      if (socket.current) socket.current.disconnect();
      if (audioContextRef.current) audioContextRef.current.close();
    };
  }, []);

  // Toggle video feed on/off
  const toggleVideoFeed = () => {
    setIsVideoOn(prevState => !prevState);
  };

  // Toggle audio feed on/off
  const toggleAudioFeed = () => {
    setIsAudioOn(prevState => !prevState);
  };

  // Play a test tone for audio
  const playTestTone = () => {
    const oscillator = audioContextRef.current.createOscillator();
    oscillator.type = 'sine';
    oscillator.frequency.setValueAtTime(440, audioContextRef.current.currentTime); // A4 note
    oscillator.connect(audioContextRef.current.destination);
    oscillator.start();
    oscillator.stop(audioContextRef.current.currentTime + 1); // Play for 1 second
  };

  return (
    <div className="App">
      <header className="App-header">
        <h2>Live Video and Audio Feed</h2>

        {/* Video Toggle Button */}
        <button onClick={toggleVideoFeed}>
          {isVideoOn ? 'Turn Video Off' : 'Turn Video On'}
        </button>

        {/* Audio Toggle Button */}
        <button onClick={toggleAudioFeed}>
          {isAudioOn ? 'Turn Audio Off' : 'Turn Audio On'}
        </button>

        {/* Test Tone Button */}
        <button onClick={playTestTone}>Play Test Tone</button>

        {/* Video Feed */}
        {isVideoOn && (
          <div className="video-feed">
            <img
              src={`${flaskServerUrl}/stream`} // Video stream endpoint
              alt="Live Video Feed"
              onError={(e) => console.error("Error loading video feed:", e)}
              onLoad={() => console.log("Video feed loaded successfully")}
              crossOrigin="anonymous"
              style={{ width: '80%', border: '2px solid #333' }}
            />
          </div>
        )}

        {/* Audio Feed */}
        {isAudioOn && (
          <audio
            src={`${flaskServerUrl}/audio-stream`} // Live audio stream endpoint
            autoPlay
            controls
            onError={(e) => console.error("Error loading audio feed:", e)}
            onLoadStart={() => console.log("Audio feed loading...")}
          />
        )}

        {/* Direction Buttons */}
        <DirectionButtons />
      </header>
    </div>
  );
}

export default App;
