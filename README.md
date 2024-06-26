# Pneumothorax detection Binary Image Classification, using TensorFlow

### Definition
```
A pneumothorax is a collection of air outside the lung but within the pleural cavity. Leading to a collapsed lung
```

Pneumothorax is usually diagnosed by a radiologist on a chest x-ray, and can sometimes be very difficult to confirm. An accurate AI algorithm to detect pneumothorax would be useful in a lot of clinical scenarios. AI could be used to triage chest radiographs for priority interpretation, or to provide a more confident diagnosis for non-radiologists.

This repository contains code for creating a CNN model that detects pneumothorax in images using TensorFlow Keras. The project is centered on binary image classification, specifically differentiating between images with pneumothorax and those without.

This project was inspired by [Annalise.Ai](https://annalise.ai/).  
The current model has a Test Accuracy of 80%

## How to Run:

The dataset used to train the model can be downloaded [here](https://www.kaggle.com/datasets/volodymyrgavrysh/pneumothorax-binary-classification-task?select=small_train_data_set).

1. Download the data set.
2. Rename the data set folder to `Dataset` and put it in the `Pnuemothroax` folder.
3. Ensure all the files in the data set are under one directory. i.e. All the files within folders should be under `Dataset`
4. Run `DataHandler.py` (this may take a minute or two)
5. Run `model.py`

If you have any issues feel free to reach out to me my email is Sufyosman@gmail.com

