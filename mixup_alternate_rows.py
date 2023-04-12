# test 
import numpy as np
import cv2

# def mixup_alternate_rows(image, p=1):
#     if np.random.random() < p:
#         h, w = image.shape[:2]
#         for i in range(1, h, 2):
#             image[i, :, :], image[i-1, :, :] = image[i-1, :, :], image[i, :, :].copy()
#     return image

def mixup_alternate_rows(image, p=1):
    if np.random.random() < p:
        h, w = image.shape[:2]
        row_indices = np.random.permutation(h)
        for i in range(1, h, 2):
            image[row_indices[i], :, :], image[row_indices[i-1], :, :] = image[row_indices[i-1], :, :], image[row_indices[i], :, :].copy()
    return image

def mixup_row_groups(image, group_size=15, p=1):
    if np.random.random() < p:
        h, w = image.shape[:2]
        row_groups = np.arange(h).reshape(-1, group_size)
        np.random.shuffle(row_groups)
        for i in range(0, h, group_size):
            group_indices = row_groups[i // group_size]
            for j in range(1, group_size, 2):
                image[group_indices[j], :, :], image[group_indices[j-1], :, :] = image[group_indices[j-1], :, :], image[group_indices[j], :, :].copy()
    return image



# Load the image
image = cv2.imread('/user/HS400/da01075/coursework/CV/vehicle_reid/datasets/VeRi/image_test/0002_c002_00030600_0.jpg')

# Apply the transform
transformed_image = mixup_row_groups(image)

# Save the output image
cv2.imwrite('output_image.jpg', transformed_image)
