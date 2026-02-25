from pygltflib import GLTF2
import numpy as np
import struct


def load_glb(filename):
    """ Lädt eine GLB-Datei mit pygltflib """
    gltf=GLTF2().load(filename)
    return gltf


def get_bin_data(gltf, filename):
    """ Extrahiert die BIN-Daten aus der GLB-Datei """
    with open(filename, "rb") as f:
        glb_data=f.read()

    # 1️⃣ GLB-Header auslesen
    magic, version, length=struct.unpack_from("<III", glb_data, 0)
    if magic != 0x46546C67:  # 'glTF' Magic Number prüfen
        raise ValueError("Keine gültige GLB-Datei!")

    # 2️⃣ JSON-Länge extrahieren
    json_length, json_type=struct.unpack_from("<II", glb_data, 12)
    if json_type != 0x4E4F534A:  # 'JSON' als Magic Number
        raise ValueError("Erster Chunk ist nicht JSON!")

    # 3️⃣ Berechnung des Startpunkts der Binärdaten (4-Byte-Alignment beachten)
    json_end=20 + json_length  # 20 Bytes für Header + JSON-Chunk
    bin_start=(json_end + 3) & ~3  # Auf 4-Byte-Alignment runden

    # 4️⃣ Prüfen, ob eingebettete Binärdaten existieren
    if len(gltf.buffers) > 0 and gltf.buffers[0].uri is None:
        buffer=gltf.buffers[0]
        buffer_length=buffer.byteLength  # Tatsächliche Länge des BIN-Datenblocks

        # 5️⃣ Erstes relevantes BufferView finden
        for buffer_view in gltf.bufferViews:
            if buffer_view.buffer == 0:
                byte_offset=buffer_view.byteOffset
                byte_length=buffer_view.byteLength

                # Offset korrekt berechnen
                real_offset=bin_start + byte_offset

                # Sicherstellen, dass der Offset gültig ist
                if real_offset < 0 or (real_offset + byte_length) > len(glb_data):
                    raise ValueError(f"Ungültiger Offset: {real_offset} (Max: {len(glb_data)})")

                return glb_data[real_offset:real_offset + byte_length]

    raise ValueError("Kein eingebetteter BIN-Buffer gefunden!")


def get_bone_length(gltf, bin_data):
    """ Berechnet die Länge eines Bones ohne Child """

    if not gltf.skins:
        raise ValueError("Keine Skelette gefunden!")

    skin=gltf.skins[0]  # Erstes Skeleton
    joints=skin.joints  # Alle Bone-IDs

    if skin.inverseBindMatrices is None:
        raise ValueError("Keine inverseBindMatrices gefunden!")

    # 1️⃣ Inverse Bindematrizen auslesen
    accessor=gltf.accessors[skin.inverseBindMatrices]
    buffer_view=gltf.bufferViews[accessor.bufferView]
    byte_offset=buffer_view.byteOffset
    count=accessor.count

    # Extrahiere 4x4 Matrizen (float32)
    matrices=np.frombuffer(bin_data, dtype=np.float32, count=count * 16, offset=byte_offset).reshape(count, 4, 4)

    # 2️⃣ Suche Leaf-Bones (ohne Child)
    child_bones=set()
    for node in gltf.nodes:
        if node.children:
            child_bones.update(node.children)

    leaf_bones=[bone for bone in joints if bone not in child_bones]

    if not leaf_bones:
        raise ValueError("Keine Leaf-Bones gefunden!")

    # 3️⃣ Berechne die Länge für den ersten Bone ohne Child
    for leaf_bone in leaf_bones:
        node=gltf.nodes[leaf_bone]

        # Prüfe, ob ein Parent existiert
        parent_bone=next((i for i, n in enumerate(gltf.nodes) if n.children and leaf_bone in n.children), None)
        if parent_bone is None:
            continue  # Kein Parent → Länge nicht berechenbar

        # 4️⃣ Berechne die Distanz
        leaf_pos=matrices[leaf_bone][:3, 3]
        parent_pos=matrices[parent_bone][:3, 3]
        bone_length=np.linalg.norm(leaf_pos - parent_pos)

        print(f"Bone-ID {leaf_bone} (Parent: {parent_bone}) → Länge: {bone_length:.3f}")
        return bone_length  # Nur den ersten Bone zurückgeben

    raise ValueError("Kein passender Bone gefunden!")


# 📂 Datei laden & auslesen
filename="../webviewer/media/avatar.glb"
gltf=load_glb(filename)
bin_data=get_bin_data(gltf, filename)
bone_length=get_bone_length(gltf, bin_data)

print(f"✅ Bone-Länge: {bone_length:.3f}")

