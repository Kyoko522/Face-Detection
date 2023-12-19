import cv2
import os


video=cv2.VideoCapture(int(input('camera (0 for webcam 1 for iPhone camera): ')))

facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count=0

nameID=str(input("Enter Your Name: ")).lower()

path='image/'+nameID

isExist = os.path.exists(path)

if isExist:
	print("Name Already Taken")
	nameID=str(input('Enter Another Name: '))
else:
	os.makedirs(path)

while True:
	ret,frame=video.read()
	faces=facedetect.detectMultiScale(frame,1.3,5)
	for x,y,w,h in faces:
		count+=1
		name='./image/'+nameID+'/'+str(count) +'.jpg'
		print("Creating Images ................"+ name)
		cv2.imwrite(name, frame[y:y+h,x:x+w]) #cutting the image of the face from the picture
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,244,0),3)
	cv2.imshow("WindowFrame",frame)
	cv2.waitKey(1)
	if count > 100:     #the number of picture you want it to take
		break
video.release()
cv2.destroyAllWindows()

