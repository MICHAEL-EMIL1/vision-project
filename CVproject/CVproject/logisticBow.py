import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns

random.seed(5)
np.random.seed(5)
test_path = r"C:\Users\hikal\Downloads\Data\Data\test classification\test classification"


# read image & feature extraction using sift
def sift_extraction(folder_Classification_path, resize_shape):
    sift = cv2.SIFT_create()
    descriptors_list = []
    labels = []

    for classnumber in os.listdir(folder_Classification_path):
        category_path = os.path.join(folder_Classification_path, classnumber)
        if os.path.isdir(category_path):
            for subfolder in os.listdir(category_path):
                subfolder_path = os.path.join(category_path, subfolder)
                if os.path.isdir(subfolder_path)and subfolder == "Train":
                    for image_file in os.listdir(subfolder_path):
                        image_path = os.path.join(subfolder_path, image_file)
                        img = cv2.imread(image_path)
                        if img is None or img.dtype != 'uint8':
                            continue

                        # Convert image to grayscale
                        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                        # Resize image
                        resized_img = cv2.resize(gray_img, resize_shape)

                        _, des = sift.detectAndCompute(resized_img, None)
                        if des is not None:
                            descriptors_list.append(des)
                            labels.append(int(classnumber))

    return descriptors_list, labels


# cluster
def clusters(descriptors_list, k):
    all_descriptors = np.vstack(descriptors_list)
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(all_descriptors)
    visualWords = kmeans.cluster_centers_
    return visualWords


def visualWordDescriptor(descriptor_list, visual_words):
    assignments_list = []
    for descriptors in descriptor_list:
        distance = np.linalg.norm(descriptors[:, None] - visual_words, axis=2)
        assignment = np.argmin(distance, axis=1)
        assignments_list.append(assignment)
    return assignments_list


def histogram(assignments_list, k):
    histograms_list = []
    for assignment in assignments_list:
        histo, _ = np.histogram(assignment, bins=np.arange(k + 1), density=True)
        # histo /= np.linalg.norm(histo, ord=2)  # L2 normalization
        histograms_list.append(histo)
    return np.array(histograms_list)


# LOGISTIC


def train_logistic_model(data, labels):
    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.30, random_state=10)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Create model
    model = LogisticRegression(max_iter=1000, random_state=0)
    # train model
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    end_time = time.time()

    print(f"total training time is {end_time - start_time} seconds")
    return model, scaler


def test_logistic_model_on_validation(folder_Classification_path, visualWords, model, scaler, resize_shape):
    sift = cv2.SIFT_create()
    correct_predictions = 0
    total_images = 0

    start_time = time.time()
    for classnumber in os.listdir(folder_Classification_path):
        class_path = os.path.join(folder_Classification_path, classnumber, 'validation')
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                example_img = cv2.imread(image_path)
                if example_img is None or example_img.dtype != 'uint8':
                    print(f"Error: Unable to load or incorrect depth for {image_path}")
                    continue

                # Convert image to grayscale
                gray_example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2GRAY)

                # Resize image
                resized_example_img = cv2.resize(gray_example_img, resize_shape)

                _, example_descriptors = sift.detectAndCompute(resized_example_img, None)
                if example_descriptors is None:
                    print(f"Error: No SIFT features extracted for {image_path}")
                    continue

                example_assignments = visualWordDescriptor([example_descriptors], visualWords)
                example_histogram = histogram(example_assignments, len(visualWords))
                example_histogram_scaled = scaler.transform(example_histogram)
                predicted_class = model.predict(example_histogram_scaled)[0]

                true_class = int(classnumber)
                total_images += 1
                if predicted_class == true_class:
                    correct_predictions += 1

                print(f"Image: {image_path}, True Class: {true_class}, Predicted Class: {predicted_class}")

    accuracy = correct_predictions / total_images if total_images > 0 else 0
    end_time = time.time()
    print(f"Accuracy on Validation Set: {accuracy * 100:.2f}%")
    print(f"total validation time is {end_time - start_time} seconds")


def test_logistic_model_on_testing(test_path, visualWords, model, scaler, resize_shape):
    sift = cv2.SIFT_create()
    correct_predictions = 0
    total_images = 0

    start_time = time.time()
    for classnumber in os.listdir(test_path):
        class_path = os.path.join(test_path, classnumber)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                example_img = cv2.imread(image_path)
                if example_img is None or example_img.dtype != 'uint8':
                    print(f"Error: Unable to load or incorrect depth for {image_path}")
                    continue

                # Convert image to grayscale
                gray_example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2GRAY)

                # Resize image
                resized_example_img = cv2.resize(gray_example_img, resize_shape)

                _, example_descriptors = sift.detectAndCompute(resized_example_img, None)
                if example_descriptors is None:
                    print(f"Error: No SIFT features extracted for {image_path}")
                    continue

                example_assignments = visualWordDescriptor([example_descriptors], visualWords)
                example_histogram = histogram(example_assignments, len(visualWords))
                example_histogram_scaled = scaler.transform(example_histogram)
                predicted_class = model.predict(example_histogram_scaled)[0]

                true_class = int(classnumber)
                total_images += 1
                if predicted_class == true_class:
                    correct_predictions += 1

                print(f"Image: {image_path}, True Class: {true_class}, Predicted Class: {predicted_class}")

    accuracy = correct_predictions / total_images if total_images > 0 else 0
    end_time=time.time()
    print(f"Accuracy on testing Set: {accuracy * 100:.2f}%")
    print(f"total testing time is {end_time - start_time} seconds")


def main():
    folder_Classification_path = r"C:\Users\hikal\Downloads\Data\Data\Product Classification"
    resize_shape = (130, 130)  # Change the size as needed

    descriptors_list, labels = sift_extraction(folder_Classification_path, resize_shape)
    print(f"number of extracted descriptors:{len(descriptors_list)}")
    k = 50
    visualWords = clusters(descriptors_list, k)
    print(f"shape:{visualWords.shape}")
    assignments = visualWordDescriptor(descriptors_list, visualWords)
    histograms = histogram(assignments, k)
    # for i, histogram in enumerate(histograms):
    # plt.bar(np.arange(len(histogram)), histogram, label=f"image{i + 1}", alpha=0.5)
    # plt.plot(histogram, label=f"image{i + 1}")

    # plt.title('histograms of visual words')
    # plt.xlabel('visual word index')
    # plt.ylabel('frequency')
    # plt.legend()
    # plt.show()

    data = np.array(histograms)
    labels = np.array(labels[:len(histograms)])

    print(f"histogram:{len(assignments)}")
    print(f"Logistic regression model")

    logistic_regression_model, scaler = train_logistic_model(data, labels)

    test_logistic_model_on_validation(folder_Classification_path, visualWords, logistic_regression_model, scaler,
                                      resize_shape)
    test_logistic_model_on_testing(test_path, visualWords, logistic_regression_model, scaler, resize_shape)


if __name__ == "__main__":
    main()
