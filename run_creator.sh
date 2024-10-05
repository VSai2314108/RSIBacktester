#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Get the current date and time
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

# Run the Python script using nohup and capture the PID
nohup python3 -u FinalCombined.py > output.log 2>&1 & PID=$!

# Log the PID and timestamp
echo "$TIMESTAMP - Process started with PID $PID" >> process.log

# Print the process ID to the console
echo "Process started with PID $PID"

# Append the PID to the output log as well
echo "Process ID: $PID" >> output.log
