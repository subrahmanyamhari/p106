import cv2


# Create our body classifier
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    body_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Pass frame to our body classifier
    bodies = body_cascade.detectMultiScale(body_gray,1.1,8)
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break
    cv2.imshow("frame",frame)
cap.release()
cv2.destroyAllWindows()
