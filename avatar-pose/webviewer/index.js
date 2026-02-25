/**
 * index.js
 *
 * Authors: Ann-Kristin Schulte, Joans D. Stephan
 * License: Apache License 2.0
 *
 * Description:
 * This script initializes a Three.js scene, connects to a WebSocket server,
 * loads a GLTF 3D model, applies real-time skeletal transformations,
 * and handles user input for sending messages to the WebSocket server.
 */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const server_url = import.meta.env.VITE_SERVER_URL;

const sizes = {
    width: window.innerWidth,
    height: window.innerHeight
};

let lastRenderTime = 0;
const frameInterval = 1000 / 50;
let boneList = null

const camera_position = { x: 0, y: 2.8, z: 2 };
const canvas = document.querySelector('.webgl');
const scene = new THREE.Scene();
scene.background = new THREE.Color( 0xffffff );

// Camera setup
let camera = new THREE.PerspectiveCamera(75, sizes.width / sizes.height, 0.1, 100);
camera.position.set(camera_position.x, camera_position.y, camera_position.z);
scene.add(camera);

const ws_address = "ws://" + server_url + ":8765"
let socket = null


/**
 * Establishes a WebSocket connection and handles server messages.
 */
function connect(){

    console.log("Connecting server")
    socket = new WebSocket(ws_address);

    socket.onopen = function(e) {
        console.log('Connection to ws server established')
    };

    socket.onmessage = function(event) {

        if(boneList != null) {

            let msg = JSON.parse(event.data)

            for(const key in msg) {
                //console.log(key)
                //console.log(msg[key])

                if (msg.hasOwnProperty(key)) {
                    let i = -1;
                    for(let i2 = 0; i2 < boneList.length; i2++){

                        if(boneList[i2].name === key){
                            i = i2;
                        }
                    }
                    boneList[i].quaternion["_x"] = msg[key].rotation[0];
                    boneList[i].quaternion["_y"] = msg[key].rotation[1];
                    boneList[i].quaternion["_z"] = msg[key].rotation[2];
                    boneList[i].quaternion["_w"] = msg[key].rotation[3];
                }
            }
        }
    };

    socket.onclose = function(event) {

        if (event.wasClean) {
            console.log(`[close] Connection closed cleanly, code=${event.code}, reason=${event.reason}`);
        } else {
            console.log('[close] Connection died, attempting to reconnect...');
        }

        console.log("reconnect in 3s")
        setTimeout(connect, 3000);
    };

    socket.onerror = function(error) {

        console.error(`[error] ${error.message}`);
        socket.close();

        console.log("reconnect in 3s")
        setTimeout(connect, 3000);
    };
}

connect();

// Load GLTF model
const loader = new GLTFLoader();
let scale_factor = 1.8;

loader.load('media/avatar.glb', function (gltf) {
    gltf.scene.scale.set(scale_factor, scale_factor, scale_factor);
    scene.add(gltf.scene);
    console.log(gltf.scene);

    gltf.scene.traverse(function(child) {
    if (child.isSkinnedMesh) {

        const skeleton = child.skeleton;
        boneList = skeleton.bones;
    }
  });

}, undefined, function (error) {
    console.error(error);
});

// Load background texture
const texture_loader = new THREE.TextureLoader();
var bgTexture = texture_loader.load(
    'media/bg.jpg', // Pfad zur Textur
    function (texture) {
        // Erfolgreich geladen
        console.log('Background texture loaded successfully');

        // Erstelle ein großes Rechteck als Hintergrund
        const bgGeometry = new THREE.PlaneGeometry(160, 90); // Plane mit Größe 100x100
        const bgMaterial = new THREE.MeshBasicMaterial({ map: texture });
        const backgroundMesh = new THREE.Mesh(bgGeometry, bgMaterial);

        // Positioniere den Hintergrund hinter der Szene
        backgroundMesh.position.z = -60; // Positioniere den Hintergrund weiter hinten
        scene.add(backgroundMesh);
    },
    undefined, function (error) {
        console.error('Error loading background texture:', error);
    }
);

// Lighting setup
var light_front = new THREE.DirectionalLight('white', 1);
light_front.position.set(2, 2, 5);
scene.add(light_front);

var light_back = new THREE.DirectionalLight('white', 1);
light_back.position.set(2, 2, -5);
scene.add(light_back);

const ambientLight = new THREE.AmbientLight('white', 0.5);
scene.add(ambientLight);

// Renderer setup
const renderer = new THREE.WebGLRenderer({
    canvas: canvas
});
renderer.setSize(sizes.width, (sizes.height-75));
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;

/**
 * Renders the scene at a fixed frame rate.
 *
 * @param {number} time - The current timestamp from requestAnimationFrame.
 */
function animate(time) {
    requestAnimationFrame(animate);

    const timeSinceLastRender = time - lastRenderTime;

    if (timeSinceLastRender >= frameInterval) {
        lastRenderTime = time;
        renderer.render(scene, camera);
    }
}
animate();

// Handle user input
/**
 * Sends user input to the WebSocket server when Enter is pressed.
 */
document.addEventListener("DOMContentLoaded", () => {
    const textInput = document.getElementById("text_input");

    textInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            const inputValue = event.target.value;
            socket.send(inputValue);
            event.target.value = '';
        }
    });
});

