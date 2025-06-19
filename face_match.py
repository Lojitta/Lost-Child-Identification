import face_recognition
import os
import cv2
import numpy as np
import screeninfo
# === Load known faces (from known_faces folder) ===
known_encodings = []
known_names = []

known_dir = 'known_faces'
for file in os.listdir(known_dir):
    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(known_dir, file)
        print(f"Processing known face: {path}")

        image_bgr = cv2.imread(path)
        if image_bgr is None:
            print(f"⚠️ Could not load image: {file}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(image_rgb)

        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(file)[0])
        else:
            print(f"⚠️ No face found in {file}")

# === Load test image ===
test_image_path = r'C:\Users\Lojit\OneDrive\Desktop\lost-child-id\test_faces\final.jpg'

print(f"\nProcessing test image: {test_image_path}")
test_bgr = cv2.imread(test_image_path)
if test_bgr is None:
    print("❌ Error: Could not load test image.")
    exit()

test_rgb = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2RGB)

# Get all face locations and encodings
face_locations = face_recognition.face_locations(test_rgb)
face_encodings = face_recognition.face_encodings(test_rgb, face_locations)

if not face_encodings:
    print("❌ No face found in the test image.")
    exit()

# Compare and annotate
for i, (encoding, location) in enumerate(zip(face_encodings, face_locations)):
    results = face_recognition.compare_faces(known_encodings, encoding)
    name = "Unknown"

    for j, match in enumerate(results):
        if match:
            name = known_names[j]
            print(f"✅ Match found at face #{i+1}: {name}")
            break
    else:
        print(f"❌ No match for face #{i+1}")

    # Draw rectangle and label
    top, right, bottom, left = location
    cv2.rectangle(test_bgr, (left, top), (right, bottom), (0, 255, 0), 2)  # Green
    cv2.putText(test_bgr, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# === Resize image to fit screen ===
screen = screeninfo.get_monitors()[0]
screen_width = screen.width
screen_height = screen.height

image_height, image_width = test_bgr.shape[:2]
scale_w = screen_width / image_width
scale_h = screen_height / image_height
scale = min(scale_w, scale_h)

new_width = int(image_width * scale)
new_height = int(image_height * scale)

resized = cv2.resize(test_bgr, (new_width, new_height))
# === Display result ===
cv2.imshow("Result", test_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
