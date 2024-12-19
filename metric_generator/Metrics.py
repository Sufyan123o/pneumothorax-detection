import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import classification_report, roc_auc_score

# Define paths
data_dir = '../Data'  # Adjust if your data directory is different
test_dir = os.path.join(data_dir, 'test')

# Parameters
batch_size = 32
img_size = (224, 224)  # Same as during training

# Load the trained model
model = load_model("Custom_Pneumothorax_classification.h5")

# Prepare the test data generator
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False  # Important for correct label alignment
)
# Evaluate the model on the test set (unseen data)
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test accuracy:", test_accuracy)

# Generate predictions
test_steps = test_generator.samples // test_generator.batch_size + 1
test_generator.reset() #resets to start from first sample

#Generates predicted probabilities for each image in the test set.
#Outputs predicted probabilities for each image. Each probability represents how confident the model is that a given image belongs to the positive class
predictions = model.predict(test_generator, steps=test_steps, verbose=1) 

y_true = test_generator.classes #Retrieves that actual class lablels (ground truth)
y_pred = predictions.ravel() #Converts the array of predicted probabilities into a flat array
y_pred_binary = (y_pred > 0.5).astype(int) #If the predicition was more than 0.5 it classifies it to the positive class (1) or (0)

# Classification report includes precision, recall, F1-score, and support for each class.
print(classification_report(y_true, y_pred_binary))

# ROC AUC Score
auc_score = roc_auc_score(y_true, y_pred) #summarises model ability to distingush between the two classes
print("ROC AUC Score:", auc_score)
