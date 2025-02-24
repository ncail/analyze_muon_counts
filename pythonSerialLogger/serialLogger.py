import serial
from datetime import datetime
import os
import time
import csv
import logging
import argparse


# Function uses retry logic to open the serial port. Exits program if port cannot be opened.
def openSerialPort(port, baudRate, maxAttempts=5, retryInterval=3):
    logging.info(f"Start openSerialPort(). Trying to open serial port {port} with baud rate {baudRate}.")
    attempts = 0
    while attempts < maxAttempts:
        try:
            ser = serial.Serial(port, baudRate)
            return ser
        except serial.SerialException:
            print(f"Failed to open serial port. Retrying in {retryInterval} seconds...")
            logging.info(f"Failed to open serial port. Retrying in {retryInterval} seconds...")
            attempts += 1
            time.sleep(retryInterval)
    print(f"Error: Failed to open serial port after {maxAttempts} attempts. Exiting program.")
    logging.info(f"Error: Failed to open serial port after {maxAttempts} attempts. Exiting program.")
    exit(1)


parser = argparse.ArgumentParser()
parser.add_argument("--port")
parser.add_argument("--output")

''' ************************************************** INITIALIZE ************************************************** '''
args = parser.parse_args()

# Set output directory.
folderPath = args.output if args.output else 'data'
if not os.path.exists(folderPath):
    os.makedirs(folderPath)

# Set port info.
port = args.port if args.port else 'COM3'
baudrate = 9600

# Configure logger.
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("Program start.")

# Open serial port.
ser = openSerialPort(port, baudrate)
logging.info("Open serial port successful.")

# Generate unique file name.
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
fileName = f"data_log_{timestamp}.csv"
filePath = os.path.join(folderPath, fileName)

''' ************************************************* WRITE TO CSV ************************************************* '''
# CSV format.
# Open CSV file in write mode.
with open(filePath, 'w', newline='') as csvfile:
    logging.info(f"File {filePath} opened as CSV.")

    csvwriter = csv.writer(csvfile)

    # Write the header row.
    csvwriter.writerow(['Time_stamp', 'Event', 'Ardn_time[ms]', 'ADC[0-1023]', 'SiPM[mV]', 'Deadtime[ms]',
                        'Temp[C]'])
    csvfile.flush()

    try:
        data_start = False
        logging.info("Loop start.")
        while True:
            # Flush logger to write message asap.
            logging.getLogger().handlers[0].flush()

            # Read serial data from microcontroller.
            raw_data = ser.readline().decode('ascii', errors='ignore').strip()

            # Skip writing all serial data until DATA START is received.
            if not data_start:
                if raw_data == "DATA START":
                    data_start = True
                    logging.info("DATA START serial message received. Begin writing serial data to file.")
                    continue
                else:
                    continue

            # Split data, assuming space separation.
            split_data = raw_data.split()
            # print("raw_data: ", raw_data, "\n")
            # print("split_data: ", split_data, "\n")

            # Break if split_data is empty.
            if not split_data:
                continue

            # Extract data elements.
            if split_data[0] == 'LOG':
                logging.debug(f'Received logging message (LOG prefix) from Arduino program:\n\t{raw_data}')
            elif len(split_data) == 6:
                # print("got muon data\n")
                event, ardn_time, adc, sipm, deadtime, temp = split_data[0:]

                # Get current timestamp.
                timestampLog = str(datetime.now())

                # Write timestamp and data to the CSV file.
                csvwriter.writerow([timestampLog, event, ardn_time, adc, sipm, deadtime, temp])

                # Flush the buffer to ensure data is written immediately.
                csvfile.flush()
            else:
                logging.debug(f"Raw data from serial.readline() is invalid and was not appended to data file: "
                              f"\n\traw_data: {raw_data}"
                              f"\n\tsplit_data: {split_data}")

    except PermissionError:
        print("Error: Permission to write to file denied.")
        logging.info("Error: Permission to write to file denied.")

    # Break out of loop if communication with microcontroller is lost.
    except serial.SerialException:
        print("Error: Communication with the microcontroller has been interrupted.")
        logging.info("Error: Communication with the microcontroller has been interrupted.")

    except KeyboardInterrupt:
        print("Program interrupted by user.")
        logging.info("Program interrupted by user.")

# Close the file.
logging.info("Closing file.")
csvfile.close()
logging.info("Program end.")
