# SIFT & HOG Feature-Based Image Classification and Matching

This project explores computer vision techniques for feature extraction and classification. It implements and evaluates two popular feature descriptors, "**SIFT (Scale-Invariant Feature Transform)**" and "**HOG (Histogram of Oriented Gradients)**".

The project is divided into two main parts:
1.  "**Image Classification**": Uses "HOG" features and "SIFT" features (with a "Bag of Visual Words" model) to train "Support Vector Machine (SVM)" classifiers on the "**CIFAR-10**" dataset.
2.  "**Keypoint Matching**": Implements "SIFT" keypoint matching to find and visualize corresponding points between two images, enhanced with "**RANSAC**" for robust filtering.

---

## Features

* "**HOG Feature Extraction**" for full-image classification.
* "**SIFT Bag of Visual Words (BoVW)**" model using "K-Means" clustering for image classification.
* "**SVM Classifier Training**" and evaluation for both "HOG" and "SIFT" features.
* "**Custom SIFT Keypoint Matching**" function demonstrating "Lowe's ratio test".
* "**RANSAC Integration**" (via "scikit-image") to filter matching outliers.
* "**Visualization**" of keypoint matches and "RANSAC" inliers.

---

## Requirements

To run this project, you will need "Python 3" and the following libraries. You can install them using "pip":

```
pip install numpy
pip install scikit-learn
pip install scikit-image
pip install matplotlib
pip install tqdm
pip install opencv-python-headless
```

Or, you can install them all at once by creating a "requirements.txt" file with the following content and running "pip install -r requirements.txt":

```
numpy
scikit-learn
scikit-image
matplotlib
tqdm
opencv-python-headless
```

---

## How to Run

The project is split into two parts.

### Part 1: Image Classification (HOG & SIFT on CIFAR-10)

This part will download the "CIFAR-10" dataset, extract features, and train classifiers to evaluate their accuracy.

**1. Download and Split Data**
First, run the "load_and_split.py" script. This will fetch the "CIFAR-10" dataset from "OpenML" and save it as "cifar10.npz".

`
python load_and_split.py
`

**2. Extract Features**
Next, run the main feature extraction script. This will load "cifar10.npz", compute "HOG" and "SIFT (BoVW)" features for the training and test sets, and save the results into two new files: "hog_features.npz" and "sift_features.npz".

`
python feature_extraction.py
`

**3. Evaluate Classifiers**
Finally, run the evaluation scripts. Each script will load its corresponding ".npz" feature file, train a "Linear SVM", and print the final classification accuracy on the test set.

* **To evaluate HOG:**
    `
    python evaluate_hog.py
    `

* **To evaluate SIFT:**
    `
    python evaluate_sift.py
    `

### Part 2: Keypoint Matching (SIFT & RANSAC)

This part demonstrates qualitative "SIFT" matching between two images.

**1. Add Images**
This script (`sift_matching.py`) expects two images named "**bobbie1.JPG**" and "**bobbie_template.JPG**" to be in the same directory. (You will need to provide these images yourself).

**2. Run the Matching Script**
Once the images are in place, simply run the script:

`
python sift_matching.py
`

This will open several "matplotlib" windows showing:
* The keypoints detected in each image.
* The raw keypoint matches found using the custom matching function.
* The refined, robust matches (inliers) found after applying "RANSAC".

---

## Project File Structure

* `load_and_split.py`: Downloads and prepares the "CIFAR-10" dataset into "cifar10.npz".
* `feature_extraction.py`: Loads `cifar10.npz`, extracts "HOG" and "SIFT (BoVW)" features, and saves them to `hog_features.npz` and `sift_features.npz`.
* `evaluate_hog.py`: Loads `hog_features.npz`, trains an "SVM", and reports accuracy.
* `evaluate_sift.py`: Loads `sift_features.npz`, trains an "SVM", and reports accuracy.
* `sift_matching.py`: Performs "SIFT" matching and "RANSAC" on two local images (`bobbie1.JPG`, `bobbie_template.JPG`).
* `create_hog.py`: A helper file defining a function to create a "HOG" descriptor (note: the main pipeline in `feature_extraction.py` uses the `skimage.feature.hog` implementation).

---

## Results

This summary outlines the performance of the "HOG" and "SIFT" feature sets on the "CIFAR-10" test data when classified with a "Linear SVM".

### 1. Number of Features Extracted

* **SIFT (Bag of Visual Words, vocab_size=100)**:
    * TRAIN: "248,285" total raw descriptors (clustered into "100" visual words).
    * TEST: "55,916" total raw descriptors.
* **HOG (8x8 pixels/cell, 2x2 cells/block)**:
    * TRAIN: "5,184,000" total feature dimensions ("10,000" images * "5184" features/image).
    * TEST: "1,296,000" total feature dimensions ("4,000" images * "5184" features/image).

### 2. Classifier Performance

* **HOG**:
    * Number of Correct Matches: "1,634"
    * Total Test Samples: "4,000"
    * **Accuracy: 40.85%**
* **SIFT**:
    * Number of Correct Matches: "1,087"
    * Total Test Samples: "3,994" (Note: "6" test images failed "SIFT" extraction)
    * **Accuracy: 27.22%**

---

## Discussion

### Q1: Describe a process for performing keypoint matching using HOG features.

"Histogram of Oriented Gradients (HOGs)" are normally computed on images as a whole rather than just on keypoints. However, we can still perform keypoint matching using "HOG" features by extracting "HOG" features from the keypoints themselves.

So we can start by using "SIFT" to get keypoints. Then, instead of computing the "HOG" for the entire image, we can compute the "HOG" for a small region (e.g., a "16x16" or "32x32" pixel patch) around each keypoint. This local "HOG" descriptor can then be used as the feature vector for that keypoint. Once we have "HOG" descriptors for all keypoints in two images, we can use the same matching process as "SIFT" (e.g., nearest neighbor distance ratio) to find corresponding pairs. Then we can use "RANSAC" to filter out incorrect matches.

### Q2: Give your own interpretation of the results. Why do you think one feature set performed better than the other?

"HOG (40.85% accuracy)" performed significantly better than "SIFT (27.22% accuracy)" for the task of image classification on "CIFAR-10".

"HOG" extracts a much larger number of features than "SIFT". This alone means that "HOG" will have a better accuracy than "SIFT". "HOG" works with the entire image rather than working on keypoints. This means that "HOG" has more material to go off of.

"HOG" is also much faster than "SIFT" because it doesn't have to compute keypoints. "SIFT" is slow and less accurate because it has to compute keypoints. This would make "SIFT" more accurate for matching things (like in "sift_matching.py") rather than classifying things.
