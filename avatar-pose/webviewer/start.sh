#!/bin/bash

# 1️⃣ Ensure the SERVER_URL environment variable is set
if [ -z "$SERVER_URL" ]; then
  echo "Error: The SERVER_URL environment variable is not set."
  exit 1
fi

# 2️⃣ Create the .env file with SERVER_URL
echo "VITE_SERVER_URL=$SERVER_URL" > .env
echo ".env file created with VITE_SERVER_URL=$SERVER_URL"

# 3️⃣ Start the Vite server
npx vite --host
