import numpy as np
import scipy.ndimage as ndi
from skimage import filters
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import SIFT



# TODO: Create feature processing functions for SIFT and HOG

def match_sift_descriptors(desc1, desc2, ratio_threshold=0.75):
    matches = []
    for i, d1 in enumerate(desc1):
        distances = np.linalg.norm(desc2 - d1, axis=1)
        sorted_idx = np.argsort(distances)
        # Ensure there are at least two descriptors to compare
        if len(distances) > 1 and distances[sorted_idx[0]] < ratio_threshold * distances[sorted_idx[1]]:
            matches.append((i, sorted_idx[0]))
    return matches


def create_hog(angles, grad_norms, center, window_size=16):
    # Create HOG vector for a given window
    window_half = int(window_size / 2)
    window_angles = angles[center[0] - window_half:center[0] + window_half,
                    center[1] - window_half:center[1] + window_half]
    window_norms = grad_norms[center[0] - window_half:center[0] + window_half,
                   center[1] - window_half:center[1] + window_half]

    # Normalize angles
    window_angles[window_angles < 0] += np.pi


    # Remove NaN values
    window_angles = np.nan_to_num(window_angles)
    window_norms = np.nan_to_num(window_norms)

    grid_size = window_half
    hog_vector = np.array([])

    for i in range(2):
        for j in range(2):
            cell_angles = window_angles[i * grid_size:i * grid_size + grid_size,
                          j * grid_size:j * grid_size + grid_size]
            cell_norms = window_norms[i * grid_size:i * grid_size + grid_size,
                         j * grid_size:j * grid_size + grid_size]

            hist, _ = np.histogram(cell_angles, bins=9, range=(0, np.pi), weights=cell_norms)
            hog_vector = np.append(hog_vector, hist)


    norm = np.linalg.norm(hog_vector)
    if norm != 0:
        hog_vector = hog_vector / norm
        hog_vector[hog_vector > 0.2] = 0.2
        norm = np.linalg.norm(hog_vector)
        hog_vector = hog_vector / norm if norm != 0 else hog_vector

    return hog_vector

def calculate_grad_norms(image, sigma=1.6):
    dx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)

    dy = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]], dtype=np.float32)

    image_gaussian = filters.gaussian(image, sigma=sigma)
    img_dx = ndi.correlate(image_gaussian, dx)
    img_dy = ndi.correlate(image_gaussian, dy)
    grad_norms = np.sqrt(img_dx ** 2 + img_dy ** 2)
    angle_img = np.arctan2(img_dy, img_dx)

    return grad_norms, angle_img


def process_images(images):
    features_list = []

    for x in images:
        image_arr = np.asarray(x).reshape((32, 32, 3))
        gray_image = rgb2gray(image_arr)
        sift = SIFT()
        sift.detect_and_extract(gray_image)
        keypoints1 = sift.keypoints
        descriptors1 = sift.descriptors

        h, w = gray_image.shape
        ratio = w / h
        new_h = 300
        new_w = int(new_h * ratio)

        resized_image = resize(gray_image, (new_h, new_w))

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

    # Extract features from the training data
    X_train = data["X_train"].astype(np.uint8)
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Extract features from the testing data
    train_features = process_images(X_train)
    test_features = process_images(X_test)

    # Save the extracted features to a file
    np.savez("features_hog.npz", X_train_features=train_features, y_train=y_train,
             X_test_features=test_features, y_test=y_test)