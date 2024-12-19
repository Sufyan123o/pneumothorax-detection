import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Load your trained model
model = tf.keras.models.load_model("Custom_Pneumothorax_classification.h5")

# Define the path to the image you want to visualize
img_path = ''

# Function to load and preprocess the image
def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch axis
    img_array = img_array / 255.0  # Normalize the image if needed
    return img_array

# Identify the last convolutional layer
last_conv_layer_name = 'conv5_block3_out' # Update this based on your model's layers

# Create a model that maps the input image to the activations of the last conv layer
grad_model = tf.keras.models.Model(
    [model.inputs],
    [model.get_layer(last_conv_layer_name).output, model.output]
)

# Function to compute the Grad-CAM heatmap
def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = 0  # Assuming binary classification with sigmoid activation
        class_channel = predictions[:, pred_index]
    # Compute the gradients of the class output value with respect to the feature map
    grads = tape.gradient(class_channel, conv_outputs)
    # Pool the gradients over all the axes leaving out the channel dimension
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    # Weigh the output feature map by the pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    # Normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Generate the heatmap
img_array = get_img_array(img_path, size=(224, 224))
heatmap = make_gradcam_heatmap(img_array, grad_model)

# Function to superimpose the heatmap on the image
def display_gradcam(img_path, heatmap, alpha=0.4):
    # Load the original image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    # Apply the heatmap to the image
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    # Display the image
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()

# Display the Grad-CAM
display_gradcam(img_path, heatmap)
