#Importing Opencv library for computer vision
from cv2 import cv2

#Cascading the pre-trained classifier trained by Intel Corp.
face_cascade = cv2.CascadeClassifier('pre_trained_face_detection_model.xml')

#Reading the image for face detection
img = cv2.imread('my_image.jpg')

#Code for detecting faces in the image
'''
Parameters - 
ReadImage,
scaleFactor(more details are in the last),
minNeighbours(more details are in the last)
'''
faces = face_cascade.detectMultiScale(img, 1.1, 4)

#Code for putting rectangle around the detected faces in the image
for (x, y, w, h) in faces: 
  cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

#Create new image with rectangle on the faces
cv2.imwrite("face_detected.png", img)


#ELLABORATED DESCRIPTION
'''
scaleFactor - Basically, the scale factor is used to create your scale pyramid. More explanation,
              your model has a fixed size defined during training, which is visible in the XML. 
              This means that this size of the face is detected in the image if present. However,
              by rescaling the input image, you can resize a larger face to a smaller one, making
              it detectable by the algorithm.
              OR
              Parameter specifying how much the image size is reduced at each image-
              scale(1.1 means 10% reduction every time)

minNeighbours - it's used to control the number of false positives and false negatives. It defines
              the minimum number of positive rectangles (detect facial features) that need to be
              adjacent to a positive rectangle in order for it to be considered actually positive.
              If minNeighbors is set to 0, the slightest hint of a face will be counted as a
              definitive face, even if no other facial features are detected near it.

'''




