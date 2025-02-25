import numpy as np
from sklearn.svm import LinearSVC
# from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

try:
    # Load the npz file
    data = np.load("sift_features.npz", allow_pickle=True)
except FileNotFoundError:
    print("Error: 'hog_features.npz' not found. Please check the file path.")
    exit()
except KeyError as e:
    print(f"Error: The expected key {e} is not found in the npz file. Check available keys with data.files")
    exit()

# Extract train/test sets
X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

# Convert the 1D arrays of feature vectors to 2D arrays
X_train = np.vstack(X_train)
X_test = np.vstack(X_test)

print(f"X_test shape: {X_test.shape}")
print(f"X_train shape: {X_train.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"y_train shape: {y_train.shape}")

# Initialize and train an SVM classifier with default parameters (except verbose)
svm = LinearSVC(random_state=42, verbose=1)  # Default C=1.0, max_iter=1000
svm.fit(X_train, y_train)

# Predict on test data
y_pred = svm.predict(X_test)


print()
print()

correct_matches = np.sum(y_pred == y_test)
print(f"Number of correct matches: {correct_matches}")
total_samples = len(y_test)
print(f"Total test samples: {total_samples}")
print(f"Accuracy (computed manually): {correct_matches / total_samples * 100:.4f}%")

