import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
import os

# Load the trained model
model_path = "Custom_Pneumothorax_classification.h5"  # Path to the saved model
model = load_model(model_path)

# Directory paths
data_dir = 'Data'
test_dir = os.path.join(data_dir, 'test')

# Test dataset preprocessing
img_size = (224, 224)
batch_size = 32

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False
)

# Generate predictions for the test set
test_steps = test_generator.samples // test_generator.batch_size + 1
test_generator.reset()  # Ensure the generator starts from the first sample
y_pred = model.predict(test_generator, steps=test_steps, verbose=1)  # Predicted probabilities
y_true = test_generator.classes  # True labels

# Flatten predictions
y_pred = y_pred.ravel()  # Flatten to a 1D array for compatibility with sklearn metrics

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=2, label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_pred)

# Plot PR Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label='PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall (PR) Curve')
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.show()
