import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import imgaug.augmenters as iaa
import cv2

'''
Seperates dataset into two classes, No Pnuemothorax & Pneumothorax
'''

#If image is 1 pnuemothorax is present, 0 if not.
df = pd.read_csv('Dataset/train_data.csv')

# Create directories for storing images
no_pneumo_dir = 'Data/No Pnuemothorax'
pneumo_dir = 'Data/Pnuemothorax'

#Based off the CSV segregate images and store them in seperate directories
os.makedirs(no_pneumo_dir, exist_ok=True)
os.makedirs(pneumo_dir, exist_ok=True)

# Iterate over each row of the dataframe and move the corresponding image to the corresponding folder
for index, row in df.iterrows(): #index is the current row, then we inspect the target column
    file_name = row['file_name']
    target = row['target']
    src_path = os.path.join('Dataset', file_name)

    if target == 0:
        dst_path = os.path.join(no_pneumo_dir, file_name)
        shutil.copy(src_path, dst_path)
    elif target == 1:
        dst_path = os.path.join(pneumo_dir, file_name)
        shutil.copy(src_path, dst_path)


'''
Splits data into training, validation and testing sets.
'''

# Define data directories to store training set and validation set
data_dir = 'Data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(val_dir):
    os.makedirs(val_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)
    
#iterate each class in the main data directory
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    
    if class_name in ['train', 'val','test']:
        continue

    if os.path.isdir(class_dir):
        #define new paths for validation and training
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        
        if not os.path.exists(train_class_dir):
            os.makedirs(train_class_dir)
        if not os.path.exists(val_class_dir):
            os.makedirs(val_class_dir)       
        if not os.path.exists(test_class_dir):
            os.makedirs(test_class_dir)
        
        #create a list with all the .png files in the directory 'class_dir'
        image_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.png')]
        
        # Debugging: Print the number of images found
        print(f"Class: {class_name}, Number of images found: {len(image_paths)}")

        # Check if there are any images before splitting
        if len(image_paths) == 0:
            print(f"No images found in class {class_name}. Skipping this class.")
            continue
        
        #split images into training and validation sets (80% training, 20% validation)
        train_paths, val_paths = train_test_split(image_paths, test_size=0.2)
        train_paths, test_paths = train_test_split(image_paths, test_size=0.3)
        
        #puts training images in directory
        for path in train_paths:
            filename = os.path.basename(path)
            dst = os.path.join(train_class_dir, filename)
            shutil.copyfile(path, dst)
        
        #puts validation images in directory
        for path in val_paths:
            filename = os.path.basename(path)
            dst = os.path.join(val_class_dir, filename)
            shutil.copyfile(path, dst)
        
        #puts test images in directory
        for path in test_paths:
            filename = os.path.basename(path)
            dst = os.path.join(test_class_dir, filename)
            shutil.copyfile(path, dst)

'''
Augments Data
'''

#Define augmentation techniques
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  #flip horizontally probability 0.5
    iaa.GaussianBlur(sigma=(0, 3.0)),  #add Gaussian blur
    iaa.Affine(rotate=(-45, 45))  #rotate -45 and 45 degrees
])


input_folder = 'Data/No Pnuemothorax'

for filename in os.listdir(input_folder):
    img = cv2.imread(os.path.join(input_folder, filename))
    # Apply the augmentation sequence to the image
    augmented_img = seq(image=img)
    #save the augmented image to same folder with a new filename
    cv2.imwrite(os.path.join(input_folder, 'aug_' + filename), augmented_img)    


input_folder = 'Data/Pnuemothorax'

for filename in os.listdir(input_folder):
    img = cv2.imread(os.path.join(input_folder, filename))
    augmented_img = seq(image=img)
    cv2.imwrite(os.path.join(input_folder, 'aug_' + filename), augmented_img)