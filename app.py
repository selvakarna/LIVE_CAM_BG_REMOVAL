import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import tempfile
import os

st.set_page_config(page_title="Live Background Replacer", layout="centered")
st.title("🎥 Live Webcam Background Remover")
st.write("Upload a background image and click 'Start Camera' to apply the effect live.")

# Upload background image
bg_file = st.file_uploader("Upload Background Image", type=["jpg", "jpeg", "png"])

if bg_file:
    # bg_image = Image.open(bg_file).convert("RGBA")
    # bg_temp_path = os.path.join(tempfile.gettempdir(), "background.jpg")
    # bg_image.save(bg_temp_path)
    bg_image = Image.open(bg_file).convert("RGBA")
    bg_rgb = bg_image.convert("RGB")  # Convert to RGB to avoid JPEG transparency error
    bg_temp_path = os.path.join(tempfile.gettempdir(), "background.jpg")
    bg_rgb.save(bg_temp_path)

    if st.button("▶️ Start Camera with Effect"):
        st.success("Launching webcam window. Press 'q' to quit or 's' to save a snapshot.")

        cap = cv2.VideoCapture(0)
        print("Webcam started. Press 's' to save image, 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb).convert("RGBA")

            try:
                no_bg = remove(pil_image)
            except Exception as e:
                print("Background removal failed:", e)
                continue

            # Use uploaded background
            bg_image = Image.open(bg_temp_path).convert("RGBA")
            bg_resized = bg_image.resize(no_bg.size)

            # Composite image
            final = Image.alpha_composite(bg_resized, no_bg)
            final_bgr = cv2.cvtColor(np.array(final), cv2.COLOR_RGBA2BGR)

            # Show side-by-side
            frame_resized = cv2.resize(frame, final_bgr.shape[:2][::-1])
            combined = np.hstack((frame_resized, final_bgr))
            cv2.imshow("Original (Left) | Stylized (Right)", combined)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = "snapshot_result.jpg"
                cv2.imwrite(filename, final_bgr)
                print(f"✅ Snapshot saved as {filename}")

        cap.release()
        cv2.destroyAllWindows()
