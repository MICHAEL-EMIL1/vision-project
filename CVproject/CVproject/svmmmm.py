from random import seed
import cv2
import os
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import BaggingClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import time
from project import y_train, preprocess_image, hog_extraction


def train_model(bagging_classifier, X_train_scaled):
    training_start_time = time.time()
    bagging_classifier.fit(X_train_scaled, y_train)
    training_time = time.time() - training_start_time
    print(f"Training Time: {training_time:.2f} seconds")


def test_model(bagging_classifier, X_test_scaled):
    testing_start_time = time.time()
    predictions = bagging_classifier.predict(X_test_scaled)
    testing_time = time.time() - testing_start_time
    print(f"Testing Time: {testing_time:.2f} seconds")
    return predictions


def validate_model(bagging_classifier, scaler, max_length, folder_Classification_path):
    correct_predictions = 0
    total_images = 0
    start_time = time.time()
    for classnumber in os.listdir(folder_Classification_path):
        class_path = os.path.join(folder_Classification_path, classnumber, 'validation')
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                preprocessed_image = preprocess_image(image_path)
                hog_features, _ = hog(preprocessed_image, orientations=9, pixels_per_cell=(8, 8),
                                      cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
                if hog_features is not None:
                    hog_features = np.pad(hog_features, (0, max_length - len(hog_features)))
                    hog_features = np.array([hog_features])
                    hog_features_scaled = scaler.transform(hog_features)
                    predicted_class = bagging_classifier.predict(hog_features_scaled)[0]
                    true_class = int(classnumber)
                    total_images += 1
                    if predicted_class == true_class:
                        correct_predictions += 1
                    print(f"Image: {image_path}, True Class: {true_class}, Predicted Class: {predicted_class}")
    end_time = time.time()
    validation_time = end_time - start_time
    print(f"Testing Time: {validation_time:.2f} seconds")


def validate_model_test(bagging_classifier, scaler, max_length, test_path):
        correct_predictions = 0
        total_images = 0
        start_time = time.time()
        for classnumber in os.listdir(test_path):
            class_path = os.path.join(test_path, classnumber)
            if os.path.isdir(class_path):
                for image_file in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_file)
                    preprocessed_image = preprocess_image(image_path)
                    hog_features, _ = hog(preprocessed_image, orientations=9, pixels_per_cell=(8, 8),
                                          cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
                    if hog_features is not None:
                        hog_features = np.pad(hog_features, (0, max_length - len(hog_features)))
                        hog_features = np.array([hog_features])
                        hog_features_scaled = scaler.transform(hog_features)
                        predicted_class = bagging_classifier.predict(hog_features_scaled)[0]
                        true_class = int(classnumber)
                        total_images += 1
                        if predicted_class == true_class:
                            correct_predictions += 1
                        print(f"Image: {image_path}, True Class: {true_class}, Predicted Class: {predicted_class}")
        end_time = time.time()
        validation_time = end_time - start_time
        print(f"Testing Time: {validation_time:.2f} seconds")


def calculate_accuracy(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy:{accuracy * 100: .2f}%")


def main():
    # Set random seed for reproducibility
    seed(42)

    resize_shape = (100, 100)
    folder_Classification_path = r"C:\Users\hikal\Downloads\Data\Data\Product Classification"
    test_path = r"C:\Users\hikal\Downloads\Data\Data\Test Samples Classification"  # Add the path to your test folder

    features, labels = hog_extraction(folder_Classification_path)

    max_length = max(len(f) for f in features)
    features = [np.pad(f, (0, max_length - len(f))) for f in features]
    features = np.array(features)
    print(f"Number of extracted descriptors: {len(features)}")
    print(f"Number of labels: {len(labels)}")

    # Convert lists to arrays for easier manipulation
    features = np.array(features)
    labels = np.array(labels)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=0)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create SVM classifier and use Bagging for better generalization
    svm_classifier = SVC(kernel='linear')
    bagging_classifier = BaggingClassifier(base_estimator=svm_classifier, n_estimators=10, random_state=0)

    # Train the classifier
    train_model(bagging_classifier, X_train_scaled, y_train)

    # Predict on the test set
    predictions = test_model(bagging_classifier, X_test_scaled)

    # Calculate accuracy
    calculate_accuracy(y_test, predictions)

    # Validation on the training set
    validate_model(bagging_classifier, scaler, max_length, folder_Classification_path)

    # Validation on the separate test set
    validate_model_test(bagging_classifier, scaler, max_length, test_path)

    # Testing on the separate test set
    test_data, test_labels = hog_extraction(test_path)
    test_max_length = max(len(f) for f in test_data)
    test_features = [np.pad(f, (0, test_max_length - len(f))) for f in test_data]
    test_features = np.array(test_features)

    # Feature scaling for the test set
    test_features_scaled = scaler.transform(test_features)

    # Predict on the test set
    test_predictions = test_model(bagging_classifier, test_features_scaled)

    # Display predictions on the test set
    for i, prediction in enumerate(test_predictions):
        print(f"Test Image {i + 1}: Predicted Class: {prediction}")

    # You can further analyze the results as needed

if __name__ == "__main__":
    main()



