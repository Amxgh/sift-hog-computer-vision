from lib2to3.pgen2.tokenize import group

import numpy as np
import scipy.ndimage as ndi
from skimage import filters
from skimage.color import rgb2gray
from skimage.transform import resize


# TODO: Create feature processing functions for SIFT and HOG

def process_images(images):
    features_list = []

    for x in images:
        image_arr = np.asarray(x).reshape((32, 32, 3))
        gray_image = rgb2gray(image_arr)

        h, w = gray_image.shape
        ratio = w / h
        new_h = 300
        new_w = int(new_h * ratio)

        resized_image = resize(gray_image, (new_h, new_w))

        # TODO: Call HOG
        grad_norms, angles = calculate_grad_norms(resized_image)

        # Extract features for multiple window positions
        for i in range(0, resized_image.shape[0] - 16, 16):
            for j in range(0, resized_image.shape[1] - 16, 16):
                features = create_hog(angles, grad_norms, (i + 8, j + 8))
                features_list.append(features)

    return np.array(features_list)


if __name__ == "__main__":
    # Load the pre-split data
    data = np.load("cifar10.npz", allow_pickle=True)

    # TODO: Extract features from the training data
    X_train = data["X_train"].astype(np.uint8)
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]


    # TODO: Extract features from the testing data
    train_features = process_images(X_train)
    test_features = process_images(X_test)

    # TODO: Save the extracted features to a file

    np.savez("features_hog.npz", X_train_features=train_features, y_train=y_train,
             X_test_features=test_features, y_test=y_test)