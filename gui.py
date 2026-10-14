import cv2
import numpy as np
import tensorflow as tf
from tkinter import Tk, Label, Button, filedialog, Frame
from PIL import Image, ImageTk

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

# GUI setup
root = Tk()
root.title("TB Detector using Lung X-rays")
root.geometry("600x600")
root.configure(bg="#f2f2f2")

frame = Frame(root, bg="#f2f2f2")
frame.pack(pady=20)

label = Label(root, text="Select an X-ray Image", font=("Arial", 16), bg="#f2f2f2")
label.pack(pady=10)

img_label = Label(root, bg="#f2f2f2")
img_label.pack(pady=10)

result_label = Label(root, text="", font=("Arial", 18, "bold"), bg="#f2f2f2")
result_label.pack(pady=20)

def open_image():
    path = filedialog.askopenfilename(
        title="Select X-ray Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not path:
        return

    # Display selected image
    img = Image.open(path)
    img = img.resize((300, 300))
    tk_img = ImageTk.PhotoImage(img)
    img_label.config(image=tk_img)
    img_label.image = tk_img

    # Predict TB / Healthy
    x = preprocess_image(path)
    pred = model.predict(x, verbose=0)[0][0]
    confidence = float(pred) * 100
    label_text = "Tuberculosis" if pred > 0.5 else "Healthy"
    color = "red" if label_text == "Tuberculosis" else "green"

    result_label.config(
        text=f"{label_text} ({confidence:.2f}%)",
        fg=color
    )

Button(root, text="Upload X-ray", command=open_image, font=("Arial", 14),
       bg="#007BFF", fg="white", padx=10, pady=5).pack()

root.mainloop()
