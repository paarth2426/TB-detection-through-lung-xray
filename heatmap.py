import threading, traceback, os
import cv2
import numpy as np
import tensorflow as tf
from tkinter import Tk, Label, Button, filedialog, Frame
from PIL import Image, ImageTk

MODEL_PATH = "tb_detector_binary.h5"
IMG_SIZE = (128, 128)
DISPLAY_SIZE = (450, 450)
ALPHA = 0.5

model = tf.keras.models.load_model(MODEL_PATH)

def find_last_conv_layer_recursive(model):
    def recurse(layers):
        for layer in reversed(layers):
            if isinstance(layer, (tf.keras.Model, tf.keras.Sequential)):
                found = recurse(layer.layers)
                if found is not None:
                    return found
            else:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    return layer
        return None
    return recurse(model.layers)

def make_gradcam_heatmap(img_array, model, last_conv_layer):
    try:
        grad_model = tf.keras.models.Model(inputs=[model.inputs], outputs=[model.get_layer(last_conv_layer.name).output, model.output])
    except Exception as e:
        raise RuntimeError(f"Could not build grad_model: {e}")
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if predictions.shape[-1] == 1:
            class_channel = predictions[:, 0]
        else:
            class_channel = tf.reduce_max(predictions, axis=-1)
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients are None")
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def make_saliency_heatmap(img_array, model):
    img_tensor = tf.convert_to_tensor(img_array)
    img_tensor = tf.cast(img_tensor, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor)
        if preds.shape[-1] == 1:
            score = preds[:, 0]
        else:
            class_idx = tf.argmax(preds[0])
            score = preds[:, class_idx]
    grads = tape.gradient(score, img_tensor)
    if grads is None:
        raise RuntimeError("Saliency gradients are None")
    grads_abs = tf.abs(grads)
    heatmap = tf.reduce_max(grads_abs, axis=-1)[0]
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap_on_image(original_gray, heatmap, alpha=ALPHA):
    heatmap_resized = cv2.resize(heatmap, (original_gray.shape[1], original_gray.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    orig_rgb = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
    overlayed = cv2.addWeighted(heatmap_color, alpha, orig_rgb, 1 - alpha, 0)
    return overlayed

def preprocess_image_np(image_path, img_size=IMG_SIZE):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read image.")
    resized = cv2.resize(img, img_size)
    normalized = resized.astype(np.float32) / 255.0
    inp = normalized.reshape(1, img_size[0], img_size[1], 1)
    return inp, resized

root = Tk()
root.title("TB Detector with Heatmap")
root.geometry("720x760")
root.configure(bg="#f2f2f2")
frame = Frame(root, bg="#f2f2f2")
frame.pack(pady=12)
title_label = Label(root, text="Select an X-ray Image", font=("Arial", 16), bg="#f2f2f2")
title_label.pack(pady=6)
img_label = Label(root, bg="#f2f2f2")
img_label.pack(pady=8)
result_label = Label(root, text="", font=("Arial", 16, "bold"), bg="#f2f2f2")
result_label.pack(pady=6)
state = {"path": None, "orig_gray": None, "pred": None, "last_heatmap": None}

def display_pil_on_label(pil_img):
    pil_img = pil_img.resize(DISPLAY_SIZE)
    tk_img = ImageTk.PhotoImage(pil_img)
    img_label.config(image=tk_img)
    img_label.image = tk_img

def open_image():
    try:
        path = filedialog.askopenfilename(title="Select X-ray Image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        state["path"] = path
        inp, orig_gray = preprocess_image_np(path)
        state["orig_gray"] = orig_gray
        pred = model.predict(inp, verbose=0)[0]
        if isinstance(pred, (list, np.ndarray)):
            pred_val = float(pred[0]) if np.shape(pred)[0] == 1 else float(np.max(pred))
        else:
            pred_val = float(pred)
        state["pred"] = pred_val
        confidence = pred_val * 100.0
        label_text = "Tuberculosis" if pred_val > 0.5 else "Healthy"
        color = "red" if label_text == "Tuberculosis" else "green"
        result_label.config(text=f"{label_text} ({confidence:.2f}%)", fg=color)
        pil_preview = Image.fromarray(cv2.cvtColor(cv2.resize(orig_gray, DISPLAY_SIZE), cv2.COLOR_GRAY2RGB))
        display_pil_on_label(pil_preview)
    except Exception as e:
        result_label.config(text=f"Error: {e}", fg="orange")
        print(traceback.format_exc())

def _compute_and_show_heatmap():
    try:
        if state["path"] is None:
            root.after(0, lambda: result_label.config(text="Upload an image first.", fg="orange"))
            return
        inp, orig_gray = preprocess_image_np(state["path"])
        last_conv_layer = find_last_conv_layer_recursive(model)
        heatmap = None
        if last_conv_layer is not None:
            try:
                heatmap = make_gradcam_heatmap(inp, model, last_conv_layer)
                root.after(0, lambda: result_label.config(text="Grad-CAM used", fg="black"))
            except Exception as e:
                print("Grad-CAM failed:", e)
                print(traceback.format_exc())
                heatmap = None
        if heatmap is None:
            try:
                heatmap = make_saliency_heatmap(inp, model)
                root.after(0, lambda: result_label.config(text="Saliency fallback used", fg="black"))
            except Exception as e:
                root.after(0, lambda: result_label.config(text=f"Heatmap error: {e}", fg="orange"))
                print(traceback.format_exc())
                return
        overlayed = overlay_heatmap_on_image(orig_gray, heatmap, alpha=ALPHA)
        state["last_heatmap"] = overlayed
        overlayed_rgb = cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)
        pil_overlay = Image.fromarray(overlayed_rgb)
        root.after(0, lambda: display_pil_on_label(pil_overlay))
        root.after(0, lambda: result_label.config(text="Heatmap shown", fg="black"))
    except Exception as e:
        root.after(0, lambda: result_label.config(text=f"Heatmap error: {e}", fg="orange"))
        print(traceback.format_exc())

def show_heatmap():
    t = threading.Thread(target=_compute_and_show_heatmap, daemon=True)
    t.start()

def save_heatmap():
    if state.get("last_heatmap") is None:
        result_label.config(text="No heatmap to save", fg="orange")
        return
    path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image","*.png")], title="Save heatmap as")
    if not path:
        return
    cv2.imwrite(path, state["last_heatmap"])
    result_label.config(text=f"Saved to {os.path.basename(path)}", fg="black")

Button(root, text="Upload X-ray", command=open_image, font=("Arial", 14), bg="#007BFF", fg="white", padx=10, pady=6).pack(pady=8)
Button(root, text="Show Heatmap", command=show_heatmap, font=("Arial", 14), bg="#28A745", fg="white", padx=10, pady=6).pack(pady=6)
Button(root, text="Save Heatmap", command=save_heatmap, font=("Arial", 12), bg="#6c757d", fg="white", padx=8, pady=5).pack(pady=6)
root.mainloop()
