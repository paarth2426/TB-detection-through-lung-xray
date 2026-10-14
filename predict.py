import cv2
import numpy as np
import tensorflow as tf
from tkinter import Tk, filedialog

# Load trained model
model = tf.keras.models.load_model("tb_detector_binary.h5")

# Preprocess function
def preprocess_image(image_path, img_size=(128, 128)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("❌ Could not read image. Try another one.")
    img = cv2.resize(img, img_size)
    img = img / 255.0
    img = img.reshape(1, img_size[0], img_size[1], 1)
    return img

# Open file dialog
Tk().withdraw()  # hide the root tkinter window
image_path = filedialog.askopenfilename(
    title="Select an X-ray Image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
)

if not image_path:
    print("⚠️ No image selected. Exiting.")
else:
    # Preprocess and predict
    img = preprocess_image(image_path)
    pred = model.predict(img, verbose=0)[0][0]  # get scalar value
    confidence = float(pred) * 100
    label = "Tuberculosis" if pred > 0.5 else "Healthy"

    # Show result in terminal
    print(f"✅ Prediction for {image_path}: {label} ({confidence:.2f}%)")

    # Display image with prediction text
    original = cv2.imread(image_path)
    if original is not None:
        text = f"{label} ({confidence:.1f}%)"
        color = (0, 0, 255) if label == "Tuberculosis" else (0, 255, 0)
        cv2.putText(original, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("X-ray Prediction", original)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
