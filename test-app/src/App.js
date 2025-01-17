import React, { useEffect, useRef } from 'react';
import io from 'socket.io-client';

function App() {
    const socket = useRef(null);
    const audioContextRef = useRef(null);

    useEffect(() => {
        // Initialize WebSocket connection
        socket.current = io('http://127.0.0.1:8000');

        // Initialize AudioContext
        audioContextRef.current = new (window.AudioContext || window.webkitAudinpmoContext)();

        // Listen for audio data from server
        socket.current.on('audio_data', (data) => {
            console.log("Received audio data:", data);  // Log to check if audio data is received
            const floatData = Float32Array.from(data);
            const audioBuffer = audioContextRef.current.createBuffer(1, floatData.length, 44100);
            audioBuffer.getChannelData(0).set(floatData);

            // Play the audio data
            const source = audioContextRef.current.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContextRef.current.destination);
            source.start();
        });

        return () => {
            // Cleanup on component unmount
            if (socket.current) socket.current.disconnect();
            if (audioContextRef.current) audioContextRef.current.close();
        };
    }, []);

    const playTestTone = () => {
      const oscillator = audioContextRef.current.createOscillator();
      oscillator.type = 'sine';
      oscillator.frequency.setValueAtTime(440, audioContextRef.current.currentTime); // A4 note
      oscillator.connect(audioContextRef.current.destination);
      oscillator.start();
      oscillator.stop(audioContextRef.current.currentTime + 1); // Play for 1 second
  };

    return (
        <div>
            <h1>Video Stream</h1>
            <img
                src="http://127.0.0.1:8000/stream"
                alt="Video Stream"
                style={{ width: '100%', maxWidth: '600px', height: 'auto' }}
            />
            <button onClick={playTestTone}>Play Test Tone</button>
            <p>Check the console to verify if audio data is being received.</p>
        </div>
    );
}

export default App;
