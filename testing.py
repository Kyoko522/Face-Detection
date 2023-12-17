import cv2

# read the file
img = cv2.imread("mandrill.jpg", 1)

# print the size of the image
print(img.shape)

height, width, channels = img.shape

print(height)
print(width)
print(channels)
