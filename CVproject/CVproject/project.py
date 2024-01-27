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

# Set random seed for reproducibility
seed(42)
resize_shape = (100, 100)


def train_model(bagging_classifier, X_train_scaled, y_train):
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


def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize the image
    resized_image = cv2.resize(gray_image, resize_shape)

    # Normalize pixel values to the range [0, 1]
    normalized_image = resized_image / 255.0

    # Reduce noise using GaussianBlur
    denoised_image = cv2.GaussianBlur(normalized_image, (5, 5), 0)

    return denoised_image


def hog_extraction(folder_Classification_path):
    features_list = []
    labels = []

    for classnumber in os.listdir(folder_Classification_path):
        print(f"#### Reading images classnumber {classnumber} #####")
        category_path = os.path.join(folder_Classification_path, classnumber)

        if os.path.isdir(category_path):
            for subfolder in os.listdir(category_path):
                subfolder_path = os.path.join(category_path, subfolder)

                if os.path.isdir(subfolder_path) and subfolder == "Train":
                    print(f"Subfolder: {subfolder}")
                    for image_file in os.listdir(subfolder_path):
                        image_path = os.path.join(subfolder_path, image_file)
                        print(f"Processing image: {image_path}")

                        preprocessed_image = preprocess_image(image_path)

                        # Extract HOG features from the preprocessed image
                        hog_features, _ = hog(preprocessed_image, orientations=9, pixels_per_cell=(8, 8),
                                              cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')

                        if hog_features is not None:
                            features_list.append(hog_features)
                            labels.append(classnumber)
                        else:
                            print(f"Warning: No HOG features extracted for {image_path}")

    return features_list, labels


def validation(bagging_classifier, scaler, max_length, folder_classification_path, X_test_scaled):
    # Validation
    predictions = bagging_classifier.predict(X_test_scaled)
    correct_predictions = 0
    total_images = 0

    start_time = time.time()
    for classnumber in os.listdir(folder_classification_path):
        class_path = os.path.join(folder_classification_path, classnumber, 'Validation')
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                preprocessed_image = preprocess_image(image_path)

                # Extract HOG features from the preprocessed image
                hog_features, _ = hog(preprocessed_image, orientations=9, pixels_per_cell=(8, 8),
                                      cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')

                if hog_features is not None:
                    hog_features = np.pad(hog_features, (0, max_length - len(hog_features)))
                    hog_features = np.array([hog_features])
                    hog_features_scaled = scaler.transform(hog_features)
                    predicted_class = bagging_classifier.predict(hog_features_scaled)[0]

                    true_class = int(classnumber)
                    total_images += 1
                    if int(predicted_class)  == true_class:
                        correct_predictions += 1

                    print(f"Image: {image_path}, True Class: {true_class}, Predicted Class: {predicted_class}")
    testaccuracy = correct_predictions / total_images if total_images > 0 else 0
    print(f"Accuracy on testing Set: {testaccuracy * 100:.2f}%")
    end_time = time.time()
    validation_time = end_time - start_time
    print(f"Testing Time: {validation_time:.2f} seconds")

def testfolder(bagging_classifier, scaler, max_length, test_path, X_test_scaled):
    predictions = bagging_classifier.predict(X_test_scaled)
    correct_predictions = 0
    total_images = 0
    testaccuracy = 0
    start_time = time.time()
    for classnumber in os.listdir(test_path):
        class_path = os.path.join(test_path, classnumber)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                preprocessed_image = preprocess_image(image_path)

                # Extract HOG features from the preprocessed image
                hog_features, _ = hog(preprocessed_image, orientations=9, pixels_per_cell=(8, 8),
                                      cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')

                hog_features = np.pad(hog_features, (0, max_length - len(hog_features)))
                hog_features = np.array([hog_features])
                hog_features_scaled = scaler.transform(hog_features)
                predicted_class = bagging_classifier.predict(hog_features_scaled)[0]

                true_class = int(classnumber)
                total_images += 1
                if int(predicted_class) == true_class:
                    correct_predictions += 1
                print(f"Image: {image_path}, True Class: {true_class}, Predicted Class: {predicted_class}")
    testaccuracy = correct_predictions / total_images if total_images > 0 else 0
    print(f"Accuracy on testing Set: {testaccuracy * 100:.2f}%")
    end_time = time.time()
    validation_time = end_time - start_time
    print(f"Testing Time: {validation_time:.2f} seconds")


def main():
    folder_Classification_path = r"C:\Users\hikal\Downloads\Data\Data\Product Classification"
    test_path = r"C:\Users\hikal\Downloads\Data\Data\Test Samples Classification"

    features, labels = hog_extraction(folder_Classification_path)

    max_length = max(len(f) for f in features)
    features = [np.pad(f, (0, max_length - len(f))) for f in features]
    features = np.array(features)
    print(f"Number of extracted descriptors: {len(features)}")
    print(f"Number of labels: {len(labels)}")

    # Convert lists to arrays for easier manipulation
    features = np.array(features)
    labels = np.array(labels)
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    start_time = time.time()  # Record the start time
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


    #validation(bagging_classifier, scaler, max_length, folder_Classification_path, X_test_scaled)
    #accuracy = accuracy_score(y_test, predictions)
    #print(f"Accuracy:{accuracy * 100: .2f}%")
    # tst_file(test_path)
    testfolder(bagging_classifier, scaler, max_length, test_path, X_test_scaled)


if __name__ == "__main__":
    main()