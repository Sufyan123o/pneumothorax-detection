import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

data_dir = 'Data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

#augment images
train_datagen = ImageDataGenerator(rescale=1.0/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='binary')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32, class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='binary')

#################################### CNN
inputs = Input(shape=(224, 224, 3)) #image of size 224*224 with 3 colours

#Conv2D(NoFilters,(size,size)..)
x = Conv2D(32, (3, 3), activation='relu')(inputs) #relu introduces non linearity. Allows more complex patterns
x = MaxPooling2D((2, 2))(x) #performs down sampling

x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

# x = Conv2D(256, (3, 3), activation='relu')(x)
# x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
##layers
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

train_steps_per_epoch = train_generator.samples // train_generator.batch_size
val_steps_per_epoch = val_generator.samples // val_generator.batch_size
#Train the model
model.fit(train_generator, steps_per_epoch=train_steps_per_epoch, epochs=10, validation_data=val_generator, validation_steps=val_steps_per_epoch)
model.save("Custom_Pneumothorax_classification.h5")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test accuracy:", test_accuracy)