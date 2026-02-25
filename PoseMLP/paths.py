# paths.py

"""
paths.py

Hilfsfunktionen für Dateipfade und Dateimanagement.
Stellt sicher, dass Verzeichnisse und Dateien korrekt gefunden werden (plattformübergreifend via pathlib).
"""

from pathlib import Path

def list_json_files(folder: str):
    """
    Listet alle .json Dateien in einem Verzeichnis auf.
    Wirft einen Fehler, wenn keine Dateien gefunden werden.
    """
    folder = Path(folder)
    files = sorted(folder.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .json files found in: {folder.resolve()}")
    return [str(p) for p in files]

if __name__ == "__main__":
    try:
        blender_files = list_json_files("DatasetBlender")
        print("Example file:", blender_files[0])
        print("Count:", len(blender_files))
    except FileNotFoundError as e:
        print(e)
