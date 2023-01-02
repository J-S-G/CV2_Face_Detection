import cv2

##detect variable is set to the xml file containing code face detection 
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

Photo_Img = cv2.VideoCapture('photograph.jpg')

#pixel dimension - resolution and image 
ret, img = Photo_Img.read()

# #grey variable - passing the image parameter with a grey scale detection 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# #faces variable will detect the different colored pixels in the grey scale image
# detectMultiScale detects parts of Mark's shirt if the second parameter is lower than 1.4 : GOOD = second_parameter > 1.4
# elon.jpg outlines 2 borders if second parameter is > 1.3    
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#this is taking the dimensions of the picture in the faces(images) variable
for (x, y, w, h) in faces:

    #rectangle will take 5 input parameters : image, pt1, pt2, color, thickness
    # this will create a border for the image that it is detecting 
    cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,0), 0)

#showing your image for the cv2 
cv2.imshow('img',img)

#milliseconds of time that the image file is open 0 : until it is closed
Picture_Timer = cv2.waitKey(0)

#Close image window
Photo_Img.release()

#De-allocate any associate memory usage
cv2.destroyAllWindows()