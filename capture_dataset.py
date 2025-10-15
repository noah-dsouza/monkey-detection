import cv2
import os
import time

expressions = ["expression1", "expression2", "expression3", "neutral"]
base_dir = "images"

# Create folders if they don't exist
for expr in expressions:
    os.makedirs(os.path.join(base_dir, expr), exist_ok=True)

# Setup webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("\nStarting dataset capture...")
print("This will automatically capture 50 frames per expression.\n")

for expr in expressions:
    print(f"Prepare for: {expr}")
    time.sleep(2)
    print("Recording...")

    count = 0
    while count < 50:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Capture Dataset", frame)
        filepath = os.path.join(base_dir, expr, f"{expr}_{count}.jpg")
        cv2.imwrite(filepath, frame)
        count += 1

        if cv2.waitKey(1) & 0xFF == 27:
            break

    print(f"Captured {count} images for {expr}\n")
    time.sleep(1)

print("Dataset collection complete.")
cap.release()
cv2.destroyAllWindows()
