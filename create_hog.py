import numpy as np

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
