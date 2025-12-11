import cv2
from detector import count_people_in_frame

# Load a test image
img = cv2.imread("download.jpeg")  # Make sure this image is in the same folder

# Count people
count = count_people_in_frame(img)
print(f"People detected: {count}")
