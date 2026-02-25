#!/bin/bash
#
# start.sh
#
# Authors: Ann-Kristin Schulte, Carolin Gottschalk, Jonas D. Stephan
# License: Apache License 2.0
#
# Description:
# This script automates the setup and execution of the application.
# It downloads a plugin, extracts it, installs dependencies,
# and starts the application.

# Navigate to the plugins directory
cd plugins

# Get the filename from the first script argument
FILENAME="$1"

# Download the plugin archive
wget "https://avatarpose.cobtras.com/${FILENAME}.zip"

# Extract the plugin
unzip ${FILENAME}.zip

# Navigate into the extracted plugin directory
cd ${FILENAME}

# Install required Python dependencies
pip install -r requirements.txt

# Ensure the startup script is executable
chmod 755 startup.sh

# Execute the plugin's startup script
./startup.sh

# Navigate back to the base directory
cd ..

# Start the main Python application
python3 main.py
