import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import DirectionButtons from './DirectionButtons';

function App() {
  const [ipAddress, setIpAddress] = useState('');
  const [isVideoOn, setIsVideoOn] = useState(false);
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
  }, []);

  // Toggle video feed
  const toggleVideoFeed = useCallback(() => {
    setIsVideoOn((prevState) => !prevState);
  }, []);

  // Toggle data transmission and send data
const toggleDataTransmission = useCallback(async () => {
  const data = {
    runID: "example_" + new Date().getTime(), // Unique ID for each run
    timestamp: new Date().toISOString()
  };

  try {
    const response = await fetch(`${flaskServerUrl}/save_data`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error("Failed to send data");
    }

    const responseData = await response.json();
    console.log("Data successfully sent to backend:", responseData);
  } catch (error) {
    console.error("Error sending data to backend:", error);
  }
}, [flaskServerUrl]);


  return (
    <div className="App">
      <header className="App-header">
        <h2>Live Video Feed</h2>

        {/* Video Toggle Button */}
        <button onClick={toggleVideoFeed}>
          {isVideoOn ? 'Turn Video Off' : 'Turn Video On'}
        </button>

        {/* Video Feed */}
        {isVideoOn && (
            <div className="video-feed">
              <img
                  src={`${flaskServerUrl}/stream`}  // Video stream endpoint
                  alt="Live Video Feed"
                  onError={(e) => console.error("Error loading video feed:", e)}
                  onLoad={() => console.log("Video feed loaded successfully")}
                  crossOrigin="anonymous"
                  style={{width: '80%', border: '2px solid #333'}}
              />
            </div>
        )}

        {/* Direction Buttons */}
        <DirectionButtons/>

      {/* Data Transmission Button */}
      <button onClick={toggleDataTransmission}>
        Send Data to DynamoDB
      </button>
      </header>
    </div>
  );
}

export default App;