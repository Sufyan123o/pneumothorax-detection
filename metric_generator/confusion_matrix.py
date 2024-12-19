import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the saved model
model = load_model("Custom_Pneumothorax_classification.h5")

# Set data directories
data_dir = 'Data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

# Define parameters
batch_size = 32
img_size = (224, 224)

# Create evaluation data generators
eval_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Evaluation generators with shuffle=False
eval_train_generator = eval_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False)

eval_val_generator = eval_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary',shuffle=False)

eval_test_generator = eval_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False)


# Function to generate predictions
def get_predictions(generator):
    steps = generator.samples // generator.batch_size + 1
    generator.reset()
    predictions = model.predict(generator, steps=steps, verbose=1)
    y_true = generator.classes
    y_pred_prob = predictions.ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)
    return y_true, y_pred, y_pred_prob

# Generate predictions and true labels for each dataset
y_true_train, y_pred_train, y_pred_prob_train = get_predictions(eval_train_generator)
y_true_val, y_pred_val, y_pred_prob_val = get_predictions(eval_val_generator)
y_true_test, y_pred_test, y_pred_prob_test = get_predictions(eval_test_generator)

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, dataset_type):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Pneumothorax', 'Pneumothorax'],
                yticklabels=['No Pneumothorax', 'Pneumothorax'])
    plt.title(f'Confusion Matrix - {dataset_type} Set')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Plot confusion matrices
plot_confusion_matrix(y_true_train, y_pred_train, 'Train')
plot_confusion_matrix(y_true_val, y_pred_val, 'Validation')
plot_confusion_matrix(y_true_test, y_pred_test, 'Test')

# Function to display classification report and AUC
def display_classification_report(y_true, y_pred, y_pred_prob, dataset_type):
    print(f"Classification Report - {dataset_type} Set")
    print(classification_report(y_true, y_pred))
    auc_score = roc_auc_score(y_true, y_pred_prob)
    print(f"ROC AUC Score - {dataset_type} Set: {auc_score:.4f}\n")

# Display classification reports
display_classification_report(y_true_train, y_pred_train, y_pred_prob_train, 'Training Data')
display_classification_report(y_true_val, y_pred_val, y_pred_prob_val, 'Validation Data')
display_classification_report(y_true_test, y_pred_test, y_pred_prob_test, 'Test Data')

print("Class Indices:", eval_train_generator.class_indices)
# Get filenames from the generator
filenames = eval_test_generator.filenames

# Verify a few samples
for i in range(5):
    print(f"Filename: {filenames[i]}")
    print(f"True label: {y_true_test[i]}, Predicted label: {y_pred_test[i]}")
