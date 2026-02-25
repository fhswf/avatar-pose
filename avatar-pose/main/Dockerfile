# Dockerfile
#
# Authors: Ann-Krisitn Schulte, Carolin Gottschalk, Jonas D. Stephan
# License: Apache License 2.0
#
# Description:
# This Dockerfile sets up a lightweight Python container
# with WebSockets support. It prepares the necessary directories,
# copies required files, and configures the application startup.

# Use the official Python image (version 3.12.8)
FROM python:3.12.8

# Install required Python packages
RUN pip3 install websockets

# Expose WebSocket server port
EXPOSE 8765

# Create and set up the working directory inside the container
RUN mkdir app
WORKDIR app

# Create a data directory inside the container
RUN mkdir data

# Copy necessary application files into the container
COPY main.py main.py
COPY start.sh start.sh

# Ensure the startup script is executable
RUN chmod 777 start.sh

# Define the default command to start the application
CMD ./start.sh $SignLanguageName
