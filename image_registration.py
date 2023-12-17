import cv2

# Load the pre-trained face detector model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open a connection to the camera (0 represents the default camera)(1 represent the camera on my phone)
if (int(input("camera (0-default or 1-IPhone): ")) == 0):
	cap = cv2.VideoCapture(0)
else:
	cap = cv2.VideoCapture(1)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # terminate
    if cv2.waitKey(1) & 0xFF == ord('q'): #pressing q will terminate the program
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
