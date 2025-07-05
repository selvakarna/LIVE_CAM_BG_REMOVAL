import cv2
import numpy as np
from PIL import Image
from rembg import remove

# Load new background image
bg_path = 'background.jpg'  # <-- Replace with your background image
bg_image = Image.open(bg_path).convert("RGBA")

# Open webcam
cap = cv2.VideoCapture(0)

print("Press 's' to save a snapshot, or 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL for rembg
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb).convert("RGBA")

    # Apply background removal
    try:
        no_bg = remove(pil_image)
    except Exception as e:
        print("Background removal failed:", e)
        continue

    # Resize background to match webcam frame
    resized_bg = bg_image.resize(no_bg.size)

    # Composite foreground over new background
    final = Image.alpha_composite(resized_bg, no_bg)
    final_bgr = cv2.cvtColor(np.array(final), cv2.COLOR_RGBA2BGR)

    # Show both frames side by side
    frame_resized = cv2.resize(frame, final_bgr.shape[:2][::-1])
    combined = np.hstack((frame_resized, final_bgr))

    cv2.imshow("Live - Original (Left) | Stylized (Right)", combined)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = "snapshot_result.jpg"
        cv2.imwrite(filename, final_bgr)
        print(f"✅ Snapshot saved as {filename}")

cap.release()
cv2.destroyAllWindows()