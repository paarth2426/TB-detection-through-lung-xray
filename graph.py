import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.models import load_model

# --- Load trained model ---
model = load_model("tb_detector_binary.h5")

# --- Test data setup ---
test_dir = "test"
img_size = (128, 128)
images, labels, filenames = [], [], []

# Load all images (no subfolders)
for img_name in os.listdir(test_dir):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        continue
    img_path = os.path.join(test_dir, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = cv2.resize(img, img_size) / 255.0
    images.append(img)
    filenames.append(img_name)

# Check if any images were loaded
if len(images) == 0:
    raise ValueError("❌ No valid images found in the 'test' folder.")

X_test = np.array(images).reshape(-1, 128, 128, 1)

# --- Model Predictions ---
y_pred_probs = model.predict(X_test, verbose=0).ravel()
y_pred = (y_pred_probs > 0.5).astype(int)

# --- Display predictions in terminal ---
print("\nPrediction Results:")
for name, pred, prob in zip(filenames, y_pred, y_pred_probs):
    label = "Tuberculosis" if pred == 1 else "Healthy"
    print(f"{name}: {label} ({prob*100:.2f}%)")

# --- If you don’t have true labels, create dummy ones (for graphs) ---
# This allows generating ROC/confusion plots even if ground truth unknown
y_true = np.zeros_like(y_pred)  # placeholder

# --- Confusion Matrix (using dummy labels if no ground truth) ---
cm = confusion_matrix(y_true, y_pred, labels=[0,1])

plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Healthy", "TB"])
plt.yticks(tick_marks, ["Healthy", "TB"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

# --- Classification Report (optional placeholder) ---
print("\nClassification Report (dummy labels):\n")
print(classification_report(y_true, y_pred, target_names=["Healthy", "TB"], zero_division=0))
