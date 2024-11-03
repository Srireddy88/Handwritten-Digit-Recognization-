# Handwritten Digit Recognition Using Support Vector Machines with Cross-Validation

## Abstract

This report presents a detailed analysis and implementation of a handwritten digit recognition system using the Support Vector Machine (SVM) classification technique. By leveraging the digits dataset from the scikit-learn library, the model aims to predict the numerical value of handwritten digits. The report highlights the process of data preprocessing, model training, and evaluation, emphasizing the importance of cross-validation in improving model reliability.

## 1. Introduction

Handwritten digit recognition is a critical application in the field of image processing and machine learning. Optical Character Recognition (OCR) systems, which convert images of handwritten text into machine-readable format, often utilize digit recognition techniques. The primary goal of this project is to develop a model capable of accurately identifying handwritten digits from images, specifically those in the range of 0-9.

## 2. Objectives

The objectives of this project include:

- To load and preprocess the digits dataset from the scikit-learn library.

- To train a Support Vector Machine classifier on the dataset.

- To evaluate the model's performance using various metrics such as accuracy, precision, recall, and F1 score.

- To implement cross-validation techniques to validate the model's performance and reliability.

## 3. Methodology

### 3.1 Data Description

The digits dataset contains 1797 samples of 8x8 pixel grayscale images representing handwritten digits. Each pixel intensity ranges from 0 (white) to 16 (black). The dataset is divided into training and testing subsets to facilitate model training and evaluation.

### 3.2 Data Loading

The dataset is loaded using the `datasets.load_digits()` function from the scikit-learn library. The initial exploration of the dataset reveals the shape of the data and the targets associated with each image.

```python

from sklearn import datasets

dig_data = datasets.load_digits()

```

### 3.3 Data Preprocessing

The images are reshaped into a 2D array where each row represents a flattened image with 64 pixel values. This transformation prepares the data for training the SVM classifier.

```python

digits = dig_data.images.reshape((len(dig_data.images), -1))

```

### 3.4 Splitting the Dataset

The dataset is split into training and testing subsets using a 70-30 ratio. This division ensures that the model is trained on a substantial portion of the data while retaining enough for validation.

```python

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(digits, dig_data.target, train_size=0.7, random_state=1)

```

### 3.5 Model Training

An SVM classifier is instantiated and trained using the training dataset. The model parameters are set, including `C` and `gamma`, which influence the decision boundary's complexity and the kernel's behavior.

```python

from sklearn import svm

model = svm.SVC(C=10.0, gamma=0.001)

model.fit(x_train, y_train)

```

### 3.6 Model Evaluation

The model's performance is evaluated using various metrics, including accuracy, precision, recall, and F1 score. Predictions are made on the test dataset, and the results are compared against the true labels.

```python

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, average='weighted')

recall = recall_score(y_test, y_pred, average='weighted')

f1 = f1_score(y_test, y_pred, average='weighted')

```

### 3.7 Cross-Validation

To ensure the model's robustness, 5-fold cross-validation is performed. This technique helps assess how the model will generalize to an independent dataset by training it on different subsets of the data.

```python

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, digits, dig_data.target, cv=5, scoring='accuracy')

```

## 4. Results

### 4.1 Model Performance

The model achieved an accuracy of approximately 99% on the test set. The detailed performance metrics are as follows:

- **Accuracy:** 0.99

- **Precision:** 0.99

- **Recall:** 0.99

- **F1 Score:** 0.99

### 4.2 Cross-Validation Results

The cross-validation scores indicated consistent performance across the folds, with an average score of approximately 97.2%.

## 5. Discussion

The high accuracy and other performance metrics reflect the model's ability to accurately recognize handwritten digits. The use of cross-validation provided insights into the model's reliability, indicating that it generalizes well to unseen data. Potential improvements could involve exploring different classifiers or fine-tuning the SVM parameters further.

## 6. Conclusion

The project successfully demonstrated the capability of using SVM for handwritten digit recognition, achieving an accuracy of 99% on the test dataset. The implementation of cross-validation further reinforced the model's reliability, showcasing its potential for real-world applications in digit recognition tasks.