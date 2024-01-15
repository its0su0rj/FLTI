import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import joblib

# Load the image
image = Image.open('/content/WIN_20230311_03_21_47_Pro.jpg')
image = np.array(image)

# Reshape the image to be a list of RGB values
pixels = image.reshape(-1, 3)

# Train the K-means model
# Train the K-means model
model = KMeans(n_clusters=16, n_init=10)
model.fit(pixels)



# Replace each pixel with the centroid of its cluster
compressed_pixels = model.cluster_centers_[model.labels_]
compressed_image = compressed_pixels.reshape(image.shape)

# Save the compressed image
compressed_image = Image.fromarray((compressed_image).astype(np.uint8))
compressed_image.save('compressed_image.jpg')

# Save the trained model
joblib.dump(model, 'image_compression_model.joblib')
