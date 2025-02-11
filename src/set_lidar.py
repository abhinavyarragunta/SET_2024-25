import serial
import time

def main():
    arduino = serial.Serial(port='COM3', baudrate=9600, timeout=0.1)
    # Baudrate of data output, not LiDAR

    while True:
        # bytes = read(9)

        data = arduino.readline()
        time.sleep(0.05)
        print(data)

if __name__ == "__main__":
    main()