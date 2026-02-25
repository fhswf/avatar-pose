# Dockerfile
#
# Authors: Ann-Kristin Schulte, Jonas D. Stephan
# License: Apache License 2.0
#
# Description:
# This Dockerfile sets up a lightweight Node.js container
# using Alpine Linux. It installs dependencies, copies necessary files,
# and configures the application to run with Vite.

# Use the official Node.js image (Alpine version for minimal footprint)
FROM node:23.7.0-alpine

# Expose port 5173 for the development server
EXPOSE 5173

# Create and set the working directory inside the container
RUN mkdir app
WORKDIR app

# Copy necessary application files into the container
COPY media media
COPY package.json package.json
COPY index.html index.html
COPY index.js index.js

# Install Vite, a lightweight frontend tool
RUN npm install vite

# Copy the startup script and make it executable
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Define the default command to start the application
CMD ["./start.sh"]
