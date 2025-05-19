# recognizer.py

import face_recognition
import cv2
import os

DOWNSCALE_FACTOR = 0.5
TOLERANCE = 0.45


def load_reference_encodings(folder="reference_faces"):
    known_encodings = []
    known_names = []

    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])
    return known_encodings, known_names


def calculate_confidence(face_encoding, known_encoding):
    distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
    confidence = round((1 - distance) * 100, 2)
    return confidence, distance


def recognize_faces(image_cv, known_encodings, known_names):
    try:
        small = cv2.resize(image_cv, (0, 0), fx=DOWNSCALE_FACTOR, fy=DOWNSCALE_FACTOR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        results = []
        for encoding, box in zip(face_encodings, face_locations):
            best_match = None
            best_confidence = 0
            best_distance = 1.0
            name = "Not Matched"

            for known_encoding, known_name in zip(known_encodings, known_names):
                confidence, distance = calculate_confidence(encoding, known_encoding)
                if distance < TOLERANCE and confidence > best_confidence:
                    best_confidence = confidence
                    best_distance = distance
                    name = known_name
                    best_match = True

            results.append({
                "match": best_match is not None,
                "name": name,
                "confidence": best_confidence,
                "distance": best_distance,
                "box": box
            })
        return results
    except Exception as e:
        return [{"error": str(e)}]
