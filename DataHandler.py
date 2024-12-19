import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import imgaug.augmenters as iaa
import cv2

'''
Separates dataset into two classes, No Pneumothorax & Pneumothorax
'''

# If target is 1, Pneumothorax is present; if 0, it is not.
df = pd.read_csv('Dataset/train_data.csv')

# Create directories for storing images
no_pneumo_dir = 'Data/No Pneumothorax'
pneumo_dir = 'Data/Pneumothorax'

# Segregate images and store them in separate directories based on the target column
os.makedirs(no_pneumo_dir, exist_ok=True)
os.makedirs(pneumo_dir, exist_ok=True)

for index, row in df.iterrows():  # Iterate through each row of the DataFrame
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
Splits data into training, validation, and testing sets (70%, 20%, 10%)
'''

# Define data directories for train, validation, and test sets
data_dir = 'Data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)

    if class_name in ['train', 'val', 'test']:
        continue

    if os.path.isdir(class_dir):
        # Define new paths for training, validation, and testing
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)

        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Create a list of all .png files in the directory
        image_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.png')]

        print(f"Class: {class_name}, Number of images found: {len(image_paths)}")

        if len(image_paths) == 0:
            print(f"No images found in class {class_name}. Skipping this class.")
            continue

        # First split into training and temp (30%)
        train_paths, temp_paths = train_test_split(image_paths, test_size=0.3, random_state=42)

        # Split temp into validation (20%) and testing (10%)
        val_paths, test_paths = train_test_split(temp_paths, test_size=1/3, random_state=42)

        # Put training images in the directory
        for path in train_paths:
            filename = os.path.basename(path)
            dst = os.path.join(train_class_dir, filename)
            shutil.copyfile(path, dst)

        # Put validation images in the directory
        for path in val_paths:
            filename = os.path.basename(path)
            dst = os.path.join(val_class_dir, filename)
            shutil.copyfile(path, dst)

        # Put testing images in the directory
        for path in test_paths:
            filename = os.path.basename(path)
            dst = os.path.join(test_class_dir, filename)
            shutil.copyfile(path, dst)

'''
Augments Data
'''

# # Define augmentation techniques
# seq = iaa.Sequential([
#     iaa.Fliplr(0.5),  # Flip horizontally with probability 0.5
#     iaa.GaussianBlur(sigma=(0, 3.0)),  # Add Gaussian blur
#     iaa.Affine(rotate=(-45, 45))  # Rotate between -45 and 45 degrees
# ])

# # Augment images in No Pneumothorax
# input_folder = 'Data/No Pneumothorax'
# for filename in os.listdir(input_folder):
#     img = cv2.imread(os.path.join(input_folder, filename))
#     augmented_img = seq(image=img)
#     cv2.imwrite(os.path.join(input_folder, 'aug_' + filename), augmented_img)

# # Augment images in Pneumothorax
# input_folder = 'Data/Pneumothorax'
# for filename in os.listdir(input_folder):
#     img = cv2.imread(os.path.join(input_folder, filename))
#     augmented_img = seq(image=img)
#     cv2.imwrite(os.path.join(input_folder, 'aug_' + filename), augmented_img)
