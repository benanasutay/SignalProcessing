import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'C:/Users/BENAN/Desktop/Signal/SignalProcessing/Lenna.png'  # Enter the image path here
image = cv2.imread(image_path)  

# Function to change the bit depth of the image
def change_bit_depth(image, bit_depth):
    max_val = 2**bit_depth - 1
    image = np.floor(image / (256 / (max_val + 1))).astype(np.uint8)
    image = (image * (255 / max_val)).astype(np.uint8)  # Adjust contrast
    return image

# Function to change the resolution of the image
def change_resolution(image, scale):
    height, width, _ = image.shape
    new_height = int(height * scale)
    new_width = int(width * scale)
    low_res = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    high_res = cv2.resize(low_res, (width, height), interpolation=cv2.INTER_NEAREST)
    return high_res

# Create images with different bit depths and resolutions
bit_depths = [6, 4, 2]
scales = [1, 0.5, 0.25, 0.125]

fig, axes = plt.subplots(len(scales), len(bit_depths), figsize=(12, 8))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

for i, scale in enumerate(scales):
    for j, bit_depth in enumerate(bit_depths):
        modified_image = change_bit_depth(image, bit_depth)
        modified_image = change_resolution(modified_image, scale)
        axes[i, j].imshow(cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
        axes[i, j].axis('off')
        axes[i, j].set_title(f'{bit_depth} bits\n{int(image.shape[1]*scale)}x{int(image.shape[0]*scale)}')

plt.show()