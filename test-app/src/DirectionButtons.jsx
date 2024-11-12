import React, { useEffect, useState } from 'react';
import './index.css';

function DirectionButtons() {
  const [activeDirection, setActiveDirection] = useState(null);
  const [inputMethod, setInputMethod] = useState('buttons'); // Track the selected input method

  const sendCommand = async (direction, state) => {
    try {
      const response = await fetch('http://127.0.0.1:8000/direction', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ direction, state }), // Send direction and state ('move' or 'stop')
      });
      const data = await response.json();
      console.log("Server response:", data);
    } catch (error) {
      console.error("Error sending command:", error);
    }
  };

  // Keyboard (WASD) Event Handlers
  const handleKeyDown = (event) => {
    if (inputMethod !== 'wasd') return; // Only activate if 'wasd' is selected
    let direction = null;
    switch (event.key) {
      case 'w':
      case 'W':
        direction = 'up';
        break;
      case 'a':
      case 'A':
        direction = 'left';
        break;
      case 's':
      case 'S':
        direction = 'down';
        break;
      case 'd':
      case 'D':
        direction = 'right';
        break;
      default:
        return;
    }

    if (activeDirection !== direction) {
      setActiveDirection(direction);
      sendCommand(direction, 'move');
    }
  };

  const handleKeyUp = (event) => {
    if (inputMethod !== 'wasd') return; // Only activate if 'wasd' is selected
    let direction = null;
    switch (event.key) {
      case 'w':
      case 'W':
        direction = 'up';
        break;
      case 'a':
      case 'A':
        direction = 'left';
        break;
      case 's':
      case 'S':
        direction = 'down';
        break;
      case 'd':
      case 'D':
        direction = 'right';
        break;
      default:
        return;
    }

    if (activeDirection === direction) {
      setActiveDirection(null);
      sendCommand(direction, 'stop');
    }
  };

  // Button Event Handlers
  const handleMouseDown = (direction) => {
    if (inputMethod !== 'buttons') return; // Only activate if 'buttons' is selected
    if (activeDirection !== direction) {
      setActiveDirection(direction);
      sendCommand(direction, 'move');
    }
  };

  const handleMouseUp = (direction) => {
    if (inputMethod !== 'buttons') return; // Only activate if 'buttons' is selected
    if (activeDirection === direction) {
      setActiveDirection(null);
      sendCommand(direction, 'stop');
    }
  };

  // Attach keyboard event listeners only when WASD is selected
  useEffect(() => {
    if (inputMethod === 'wasd') {
      window.addEventListener('keydown', handleKeyDown);
      window.addEventListener('keyup', handleKeyUp);
    } else {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    }

    // Clean up event listeners when input method changes
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [inputMethod, activeDirection]);

  return (
    <div>

        {/* Dropdown to select input method */}
        <select value={inputMethod} onChange={(e) => setInputMethod(e.target.value)}>
            <option value="buttons">Buttons</option>
            <option value="wasd">WASD Keys</option>
        </select>

        <div className="keyboard">

            {/* Directional buttons */}
            <button
                className={`key ${activeDirection === 'up' ? 'active' : ''}`}
                onMouseDown={() => handleMouseDown('up')}
                onMouseUp={() => handleMouseUp('up')}
            >
                Up
            </button>
            <button
                className={`key ${activeDirection === 'left' ? 'active' : ''}`}
                onMouseDown={() => handleMouseDown('left')}
                onMouseUp={() => handleMouseUp('left')}
            >
                Left
            </button>

            <button
                className={`key ${activeDirection === 'down' ? 'active' : ''}`}
                onMouseDown={() => handleMouseDown('down')}
                onMouseUp={() => handleMouseUp('down')}
            >
                Down
            </button>

            <button
                className={`key ${activeDirection === 'right' ? 'active' : ''}`}
                onMouseDown={() => handleMouseDown('right')}
                onMouseUp={() => handleMouseUp('right')}
            >
                Right
            </button>
        </div>
    </div>

  );
}

export default DirectionButtons;
