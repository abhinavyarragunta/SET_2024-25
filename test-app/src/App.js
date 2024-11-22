import React, { useState, useEffect, useRef, useCallback } from 'react';
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

  // Fetch the IP address from the Flask server
  useEffect(() => {
    const fetchIpAddress = async () => {
      try {
        const response = await fetch(`${flaskServerUrl}/get-ip`);
        const data = await response.json();
        setIpAddress(data.ip);
      } catch (error) {
        console.error("Error fetching IP:", error);
      }
    };

    fetchIpAddress();

    // Initialize WebSocket connection
    socket.current = io(flaskServerUrl);

    return () => {
      socket.current.disconnect();  // Cleanup WebSocket on component unmount
    };
  }, []);

  // Handle the WebSocket audio data reception
  useEffect(() => {
    if (!socket.current) return;

    socket.current.on('audio_data', (data) => {
      if (!isAudioOn) return;

      const floatData = Float32Array.from(data);  // Convert data to Float32Array
      const audioBuffer = audioContextRef.current.createBuffer(1, floatData.length, 44100);
      audioBuffer.getChannelData(0).set(floatData);  // Set audio buffer data

      const source = audioContextRef.current.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContextRef.current.destination);
      source.start();  // Start audio playback
    });

    return () => {
      socket.current.off('audio_data');  // Cleanup listener on unmount
    };
  }, [isAudioOn]);

  // Toggle video feed
  const toggleVideoFeed = useCallback(() => {
    setIsVideoOn((prevState) => !prevState);
  }, []);

  // Start recording audio and send it to Flask
  const startRecording = () => {
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => {
        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            const reader = new FileReader();
            reader.onload = () => {
              const audioBuffer = reader.result;
              socket.current.emit('audio_data_from_client', audioBuffer);  // Send audio data to Flask
            };
            reader.readAsArrayBuffer(event.data);
          }
        };
        mediaRecorder.start(100);  // Capture audio in chunks every 100ms
      })
      .catch(error => console.error("Error accessing the microphone:", error));
  };

  // Toggle audio feed (start/stop recording)
  const toggleAudioFeed = () => {
    setIsAudioOn(prevState => {
      const newState = !prevState;
      if (newState) {
        // Start recording when audio is turned on
        startRecording();
        // Create AudioContext if not already created or it was closed
        if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
          audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
        }
        socket.current.emit('start_audio_stream');  // Start audio stream from server
      } else {
        socket.current.emit('stop_audio_stream');  // Stop audio stream
        // Close the AudioContext if it's open
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
          audioContextRef.current.close().catch((err) => console.error('Error closing AudioContext:', err));
        }
      }

      return newState;
    });
  };

  // Play test tone
  const playTestTone = useCallback(() => {
    const oscillator = audioContextRef.current.createOscillator();
    oscillator.type = 'sine';
    oscillator.frequency.setValueAtTime(440, audioContextRef.current.currentTime);  // A4 note
    oscillator.connect(audioContextRef.current.destination);
    oscillator.start();
    oscillator.stop(audioContextRef.current.currentTime + 1);  // Play for 1 second
  }, []);

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
        <button onClick={playTestTone}>Play Test Tone (loud)</button>

        {/* Video Feed */}
        {isVideoOn && (
          <div className="video-feed">
            <img
              src={`${flaskServerUrl}/stream`}  // Video stream endpoint
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
            src={`${flaskServerUrl}/audio-stream`}  // Live audio stream endpoint
            autoPlay
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
