
# AvatarPose

AvatarPose is an application for visualizing sign language through 3D avatars in real time. It uses WebSockets for communication between the server and client, transferring quaternions and other pose parameters to a Three.js frontend. This enables dynamic adjustments of avatars based on incoming sign language gestures.


## Features

- **Real-Time Sign Language Visualization**: Animates 3D avatars based on data from sign language modules
- **WebSocket Communication**: Seamless connection between server and client for gesture and pose data transmission
- **3D Visualization**: Powered by Three.js for rendering and interacting with 3D avatars
- **GLTF Support**: Loads and processes 3D avatars in GLTF format
- **Modularity**: Easily extendable with new language modules or data sources


## Installation

1. **Use Docker to build and run**
```bash
docker build -t avatar-pose .
docker run -p 8765:8765 avatar-pose

docker build -t avatar-pose-webviewer .
docker run -e SERVER_URL="https://meinserver.de" avatar-pose-webviewer
```

2. **Start the project**
```bash
./start.sh <plugin-name>
```


## Usage

1. **Launch the Web Frontend**:
   - Open `??????` in a browser to view the 3D avatar.

2. **WebSocket Endpoints**:
   - **Data Reception (Avatar Poses)** and **Input (Sign Language Commands)**: `ws://<server-ip>:8765`

3. **Send Input Commands**:
   - Open the browser's input field.
   - Type words or sentences into the field.
   - Press `Enter` to animate the avatar with the corresponding gestures.


## License
This project is licensed under the **Apache License 2.0**.