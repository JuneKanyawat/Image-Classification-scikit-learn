import os
import pickle

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# prepare data
input_dir = 'data-clf' # directory where image data is stored
categories = ['empty', 'not_empty'] # categories of the data

data = []  # To store image data
labels = []  # To store corresponding labels
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)  # Read image
        img = resize(img, (30, 10))  # Resize image to (30, 10)
        data.append(img.flatten())  # Flatten image and store in data
        labels.append(category_idx)  # Store corresponding label index

data = np.asarray(data)  # Convert to numpy array
labels = np.asarray(labels)  # Convert to numpy array

# train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# train classifier
classifier = SVC()

parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)

# test performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)  # Predict on test data

score = accuracy_score(y_prediction, y_test)  # Calculate accuracy

print('{}% of samples were correctly classified'.format(str(score * 100)))

# Serialize and save the model
pickle.dump(best_estimator, open('../MainProject/model.p', 'wb'))