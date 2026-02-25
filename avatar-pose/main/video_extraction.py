import cv2
import os

def extract_frames(video_path, output_folder):
    # Überprüfen, ob der Ausgabepfad existiert, ansonsten erstellen
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Video öffnen
    video_capture = cv2.VideoCapture(video_path)

    # Videoinformationen abrufen
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    print(f"Frames insgesamt: {frame_count}, FPS: {fps}")

    frame_number = 0

    while True:
        success, frame = video_capture.read()

        # Abbruchbedingung, wenn keine Frames mehr gelesen werden können
        if not success:
            print("Alle Frames wurden extrahiert.")
            break

        # height, width, _ = frame.shape
        # right_half = frame[:, width // 2:]

        resized_frame = cv2.resize(frame, (3840, 1080))

        # Frame als Bild speichern
        output_file = os.path.join(output_folder, f"frame_{frame_number:04d}.jpg")
        cv2.imwrite(output_file, resized_frame)

        frame_number += 1

    # Ressourcen freigeben
    video_capture.release()
    print(f"Frames wurden in {output_folder} gespeichert.")

if __name__ == "__main__":
    video_path = "Testvideo.mp4"  # Pfad zum MP4-Video
    output_folder = "extraction"  # Ordner für die gespeicherten Frames

    extract_frames(video_path, output_folder)