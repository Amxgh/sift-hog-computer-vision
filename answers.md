# Discussion
## Instructions to run

1. Run feature_extraction.py to extract features from the training and test images.
2. Run evaluate_hog.py to evaluate the HOG features.
3. Run evaluate_sift.py to evaluate the SIFT features.

The remaining files are attached because I was not sure about the deliverables. They can be ignored if they are not
required.

## Requirements

1. The number of features extracted from each approach are as follows:
    - **SIFT**:
        - TRAIN: 248285 features.
        - TEST: 55916 features.
    - **HOG**:
        - TRAIN: 5184000 features.
        - TEST: 1296000 features.

2. The number of correct matches found:
    - **HOG**:
        - Number of Correct Matches: 1634
        - Total Test Samples: 4000
        - Accuracy: 40.85%
    - **SIFT**:
        - Number of Correct Matches: 1087
        - Total Test Samples: 3994
        - Accuracy: 27.22%

3. The accuracy of the classifiers:
    - **HOG**:
        - Accuracy: 40.85%
    - **SIFT**:
        - Accuracy: 27.22%

## Questions

1. Histogram of Oriented Gradients (HOGs) are normally computed on images as a whole rather than just on keypoints.
   However, we can still perform keypoint matching using HOG features by extracting HOG features from the keypoints
   themselves. So we can start by using SIFT to get keypoints. Then, instead of computing the HOG for the entire image,
   we can compute the HOG for a small region around each keypoint. Then we can use RANSAC to filter out incorrect
   matches.
2. HOG extracts a much larger number of features than SIFT. This alone means that HOG will have a better accuracy than
   SIFT. HOG works with the entire image rather than working on keypoints. This means that HOG has more material to go
   off of. HOG is also much faster than SIFT because it doesn't have to compute keypoints. SIFT is slow and less
   accurate because it has to compute keypoints. This would make SIFT more accurate for matching things rather than
   classifying things. 
