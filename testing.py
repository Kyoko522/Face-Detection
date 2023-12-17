import cv2

# read the file
img = cv2.imread("mandrill.jpg", 1) #the first argument that you pass in is the name of the file the second is the channel u want to pass it into
#1 loads a color image
#0 loads image in grayscale mode
#-1 loads image as such include alpha channel

# print the size of the image
print(img.shape)
height, width, channels = img.shape


print(height)       #print the height
print(width)        #print the width
print(channels)     #print the channels

print(type(img))    #print the type of the image
print(img.dtype)    #print the dtype of the image

print(img)          #print the image as a matrices

#to show an image on the screen
cv2.imshow('Mandrill', img)
k = cv2.waitKey(0) #close the file automatically
if k == 27 or k == ord('q'):
    cv2.destroyAllWindows()

#saving the image
elif k == ord('s'):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('Mandrill_grey.jpg', gray)
