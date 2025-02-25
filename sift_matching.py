import cv2

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
from skimage.feature import match_descriptors, SIFT
from skimage.color import rgb2gray, rgba2rgb
from skimage import measure
from skimage.transform import ProjectiveTransform

def convert_keypoints(sk_keypoints, size=1):
    # Convert [row, col] (i.e., [y, x]) to cv2.KeyPoint(x, y, size)
    return [cv2.KeyPoint(x=float(pt[1]), y=float(pt[0]), size=size) for pt in sk_keypoints]


def plot_key_points(image, keypoints, title):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    image_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(6, 6))
    plt.imshow(image_keypoints, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def match_sift_descriptors(desc1, desc2, ratio_threshold=0.8):
    matches = []
    for i, d in enumerate(desc1):
        distances = np.linalg.norm(desc2 - d, axis=1)
        sorted_idx = np.argsort(distances)
        if len(sorted_idx) > 1:
            closest_dist = distances[sorted_idx[0]]
            second_closest_dist = distances[sorted_idx[1]]
            ratio = closest_dist / second_closest_dist
            if ratio < ratio_threshold:
                matches.append((i, sorted_idx[0]))
    return matches

def extract_sift_features(image):
    sift = SIFT()
    sift.detect_and_extract(image)
    return sift.keypoints, np.array(sift.descriptors, dtype=np.float32)


def main():
    # Read and convert the template images (for SIFT matching)
    bobbie = cv2.imread('bobbie1.JPG')
    bobbie = cv2.cvtColor(bobbie, cv2.COLOR_BGR2RGB)
    bobbie_template = cv2.imread('bobbie_template.JPG')
    bobbie_template = cv2.cvtColor(bobbie_template, cv2.COLOR_BGR2RGB)

    # For SIFT, convert images to grayscale
    image1 = np.asarray(bobbie)
    image2 = np.asarray(bobbie_template)
    gray1 = rgb2gray(image1)
    gray2 = rgb2gray(image2)


    keypoints1, descriptors1 = extract_sift_features(gray1)
    keypoints2, descriptors2 = extract_sift_features(gray2)

    # Perform SIFT matching using the custom function
    matches = match_sift_descriptors(descriptors1, descriptors2)

    # For SIFT, convert images to grayscale
    image1 = np.asarray(bobbie)
    image2 = np.asarray(bobbie_template)

    # Resize images to have the same height for display
    height1, width1, _ = image1.shape
    height2, width2, _ = image2.shape
    target_height = max(height1, height2)

    # Compute scale factors and new widths
    scale1 = target_height / height1
    scale2 = target_height / height2
    new_width1 = int(width1 * scale1)
    new_width2 = int(width2 * scale2)

    # Resize the images using cv2.resize
    image1_resized = cv2.resize(image1, (new_width1, target_height))
    image2_resized = cv2.resize(image2, (new_width2, target_height))

    # Combine the images horizontally
    combined_image = np.hstack((image1_resized, image2_resized))

    # Convert skimage keypoints to cv2.KeyPoint objects
    cv_keypoints1 = convert_keypoints(keypoints1)
    cv_keypoints2 = convert_keypoints(keypoints2)

    plot_key_points(gray1, cv_keypoints1, "Keypoints in Image 1")
    plot_key_points(gray2, cv_keypoints2, "Keypoints in Image 2")

    # Extract matched keypoints using the indices from the matches list
    matched_points1 = np.array([keypoints1[i] for i, _ in matches])
    matched_points2 = np.array([keypoints2[j] for _, j in matches])



    # sk_M, sk_best = measure.ransac((keypoints1, keypoints2), ProjectiveTransform, min_samples=4,residual_threshold=1, max_trials=300)




    # Now use combined_image for plotting
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(combined_image)
    for match in matches:
        y1, x1 = keypoints1[match[0]]
        y2, x2 = keypoints2[match[1]]
        # Adjust the x-coordinate for the second image by adding the width of the first resized image
        ax.plot([x1, x2 + image1_resized.shape[1]], [y1, y2], 'r-')
    plt.title("SIFT Keypoint Matches")
    plt.show()



    # Now perform RANSAC on the matched keypoints
    sk_M, sk_best = measure.ransac((matched_points1, matched_points2),
                                   ProjectiveTransform,
                                   min_samples=4,
                                   residual_threshold=1,
                                   max_trials=300)

    matches = np.array(matches)

    indices_src = matches[sk_best, 1]
    src_best = np.array([keypoints2[i] for i in indices_src])[:, ::-1]

    indices_dst = matches[sk_best, 0]
    dst_best = np.array([keypoints1[i] for i in indices_dst])[:, ::-1]

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(image1_resized)
    ax2.imshow(image2_resized)

    for i in range(src_best.shape[0]):
        coordB = [dst_best[i, 0], dst_best[i, 1]]
        coordA = [src_best[i, 0], src_best[i, 1]]
        con = ConnectionPatch(xyA=coordA, xyB=coordB, coordsA="data", coordsB="data",
                              axesA=ax2, axesB=ax1, color="red")
        ax2.add_artist(con)
        ax1.plot(dst_best[i, 0], dst_best[i, 1], 'ro')
        ax2.plot(src_best[i, 0], src_best[i, 1], 'ro')

    plt.show()


if __name__ == "__main__":
    main()