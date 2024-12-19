import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.utils import class_weight
import numpy as np

# Additional evaluation metrics
from sklearn.metrics import classification_report, roc_auc_score



# Set data directories
data_dir = 'Data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

# Define parameters
batch_size = 32
img_size = (224, 224) #ResNet expected size
num_epochs = 30

# Augmentation on the training datasets
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15, #15 Degrees
    width_shift_range=0.1, #Horziontally 10%
    height_shift_range=0.1, #Vertically 10% 
    shear_range=0.01, #Mimics Distortion
    zoom_range=[0.9, 1.25], #Randomly Zooms in our out of images
    horizontal_flip=True, #Flip images horizontally (applicable for my data set)
    fill_mode='reflect' #Fills in missing pixels are transformations
)

# Preprocess Test and Validation data sets for ResNet50
test_val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)


# Load datasets into  and apply augmentations, generate batches
train_generator = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
val_generator = test_val_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary',shuffle=False)
test_generator = test_val_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False)

# get class indices for each image
classes = train_generator.classes 

#One class has more samples than the other, therefore the model becomes biased.
#Penalises misclassification of the minority class more. 
# Encounraging model to pay equal attention to both classses
class_weights = class_weight.compute_class_weight(
    class_weight='balanced', # Calculates weights based on number of samples in each class
    classes=np.unique(classes), #Finds unique class labels in the dataset
    y=classes 
)
class_weights = dict(enumerate(class_weights)) 
#Result: A dictionary where each class label is mapped to its corresponding weight.

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# ResNet50(Load image dataset, Exclude top classification layers, Image dimensions)

# Freeze the base model initially
base_model.trainable = False #Ensures only weights of the new tops layers are updated

# Add custom layers on top of the base model
inputs = base_model.input 

x = base_model.output
x = GlobalAveragePooling2D()(x) #Reduce each feature map to a single value. Capture presence of features without considering their location
x = Dense(512, activation='relu')(x) #Add layer with 512 neurons and ReLU activation
x = Dropout(0.5)(x) #Randomly set 50% of neurons to zero during each training step, prevent overfitting by promotiing independence among neurons
outputs = Dense(1, activation='sigmoid')(x) #Adds an output layer with a sigmoid function for binary classification (Output between 0,1)

model = Model(inputs=inputs, outputs=outputs) #Combine the ResNet50 model with the new custom layers into a single model

# Compile the model
#high learning rate for the initial training phase to train new layers
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), #higher learning rate to allow large adjustements so model can learn basic patterns
    loss='binary_crossentropy', #binary classification
    metrics=['accuracy'] #Track accuracy
)

# Callbacks - monitors validation loss and reduces the learning rate by a factor of -.1 if the loss doesnt improve for 3 epochs
reduce_lr = ReduceLROnPlateau(
monitor='val_loss', #Monitors val loss
    factor=0.1, #Reduce the learning rate by factor of 10
    patience=3, #Waits 3 epochs without improvement before reducing the learning rate
    verbose=1, #Prints when reduced
    min_lr=1e-7 # Sets min learning rate limit
)

#Stop training if validation loss doesnt improve for 5 epochs - Prevents overfitting by halting training if it stops improving on validation data.
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True #restores the model weights from epoch with the best val_loss
)

# Trains the custom layers on training data and validates on the validation
history = model.fit(
    train_generator,
    epochs=num_epochs, #30 Epochas
    validation_data=val_generator, #
    class_weight=class_weights, #apply class weights during training
    callbacks=[reduce_lr, early_stopping]
)



###! Train ResNet50 and Fine Tune!###

# Unfreeze ResNet50 layers for fine-tuning
base_model.trainable = True 

# Re-compile the model with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), #0.00001
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Continue training
fine_tune_epochs = 10 #10 epochs to allow resnet 50 to learn new data
total_epochs = num_epochs + fine_tune_epochs

history_fine = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1], #resumes training from the last epoch of the previous training phase.
    validation_data=val_generator,
    class_weight=class_weights, #apply class weights during training
    callbacks=[reduce_lr, early_stopping]
)

# Save the model
model.save("Custom_Pneumothorax_classificationn.h5")

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
