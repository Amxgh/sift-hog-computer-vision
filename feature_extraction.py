# Amogh Modgekar Desai - 1002060753
# Used chatgpt and GitHub copilot to complete this assignment.
import math

import PIL.Image as Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from matplotlib.patches import ConnectionPatch
from skimage import io, filters, feature
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import hog
from skimage.feature import match_descriptors, SIFT
from skimage.measure import ransac
from skimage.transform import AffineTransform, resize
from sklearn.cluster import KMeans
from tqdm import tqdm

from sift_matching import convert_keypoints, plot_key_points, match_sift_descriptors, extract_sift_features


def computer_histogram \
                (descriptors, kmeans, vocab_size):
    # If the descriptor is 1D (i.e., a single descriptor), reshape it to 2D.
    if descriptors.ndim == 1:
        descriptors = descriptors.reshape(1, -1)
    labels = kmeans.predict(descriptors)
    hist, _ = np.histogram(labels, bins=np.arange(vocab_size + 1))
    return hist


def extract_hog_features(images, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    """Extract HOG features from images."""
    hog_features = []
    for img in tqdm(images, desc="Extracting HOG features"):
        features = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block, feature_vector=True)
        hog_features.append(features)
    return np.array(hog_features)


def process_images(images):
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)
    images_gray = np.array([rgb2gray(image) for image in images])
    return images_gray


def descriptors_to_histogram(desc_list, kmeans_model):
    vocab_size = kmeans_model.n_clusters
    hist_list = []
    for desc in desc_list:
        cluster_labels = kmeans_model.predict(desc)
        # count occurrences of each cluster
        hist, _ = np.histogram(cluster_labels, bins=range(vocab_size + 1))
        hist_list.append(hist)
    return np.array(hist_list)


if __name__ == "__main__":
    # Load the pre-split CIFAR-10 data
    data = np.load("cifar10.npz", allow_pickle=True)
    X_train = data["X_train"].astype(np.uint8)
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    X_train = process_images(X_train)
    X_test = process_images(X_test)
    #
    # # Visualize the first 10 images
    # fig, axes = plt.subplots(1, 10, figsize=(10, 1))
    # for i, ax in enumerate(axes):
    #     ax.imshow(X_train[i], cmap='gray')
    #     ax.axis('off')
    # plt.show()
    # # Visualize the first 10 images
    # fig, axes = plt.subplots(1, 10, figsize=(10, 1))
    # for i, ax in enumerate(axes):
    #     ax.imshow(X_test[i], cmap='gray')
    #     ax.axis('off')
    # plt.show()
    #
    # Instantiate SIFT
    sift = SIFT()

    # Lists to store descriptors and labels for training set
    sift_features_train = []
    y_sift_features_train = []

    total_features = 0
    # Extract SIFT features from training images
    for idx in tqdm(range(X_train.shape[0]), desc="Extracting SIFT from Train"):
        try:
            # Detect and extract SIFT features
            sift.detect_and_extract(X_train[idx])
            num_features = sift.descriptors.shape[0]
            total_features += num_features
            # If descriptors were extracted successfully, store them and the label
            if sift.descriptors is not None:
                sift_features_train.append(sift.descriptors)
                y_sift_features_train.append(y_train[idx])
        except Exception as e:
            pass

    print(f"Total SIFT - TRAIN features extracted: {total_features}")

    # Convert the list of SIFT features to a numpy array
    sift_features_train_np = np.concatenate(sift_features_train)

    # Create a KMeans model to cluster the SIFT features
    vocab_size = 100
    kmeans = KMeans(n_clusters=vocab_size, random_state=42)

    # Fit the KMeans model to the SIFT features
    kmeans.fit(sift_features_train_np)

    X_train_sift = descriptors_to_histogram(sift_features_train, kmeans)

    # sift = SIFT()

    # Similarly, extract SIFT features from the test set
    sift_features_test = []
    y_sift_features_test = []

    sift2 = SIFT()
    print(f"Processing {X_test.shape[0]} images...")

    total_features = 0
    for idx in tqdm(range(X_test.shape[0]), desc="Extracting SIFT from Test"):
        try:
            sift2.detect_and_extract(X_test[idx])
            num_features = sift.descriptors.shape[0]
            total_features += num_features
            if sift2.descriptors is not None:
                sift_features_test.append(sift2.descriptors)
                y_sift_features_test.append(y_test[idx])
        except Exception as e:
            pass

    print(f"Total SIFT - TEST features extracted: {total_features}")

    X_test_sift = descriptors_to_histogram(sift_features_test, kmeans)

    y_train_sift = np.array(y_sift_features_train)
    y_test_sift = np.array(y_sift_features_test)

    np.savez(
        "sift_features.npz",
        X_train=X_train_sift,
        y_train=y_train_sift,
        X_test=X_test_sift,
        y_test=y_test_sift
    )
    print("SIFT features (Bag of Visual Words) saved to sift_features.npz!")

    # Extract HOG features
    X_train_hog = extract_hog_features(X_train)
    X_test_hog = extract_hog_features(X_test)

    # print(X_train_hog.shape)
    # Create a KMeans model to cluster the HOG features
    # vocab_size = 100  # Adjust the number of clusters as needed
    # kmeans = KMeans(n_clusters=vocab_size, random_state=42)
    # kmeans.fit(X_train_hog)
    # # #
    # # # # Convert HOG features into bag-of-words histograms
    # # # Assuming extract_hog_features returns a list of arrays,
    # # # where each array contains the HOG descriptors for one image:
    # # X_train_hog_hist = np.array([computer_histogram
    # #                              (desc, kmeans, vocab_size) for desc in X_train_hog])
    # # X_test_hog_hist = np.array([computer_histogram
    # #                             (desc, kmeans, vocab_size) for desc in X_test_hog])

    print("Number of HOG features TRAIN", X_train_hog.shape[1] * X_train_hog.shape[0])
    print("Number of HOG features TEST:", X_test_hog.shape[1] * X_test_hog.shape[0])

    # Save the extracted features
    np.savez(
        "hog_features.npz",
        X_train=X_train_hog,
        y_train=y_train,
        X_test=X_test_hog,
        y_test=y_test
    )
    print("HOG features (Bag of Visual Words) saved to hog_features.npz!")
