import cv2

# Load Haar cascade algorithm
alg = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

haar_cascade = cv2.CascadeClassifier(alg)
# open web camera
cam=cv2.VideoCapture(0)

# check camera
if not cam.isOpened():
    print("Error: Can't open camera")
    exit()
    
while True:
    # read camera frame
    ret,img = cam.read()
    
    # ret  returns True if camera is read correctly
    if not ret:
        print("Error: Can't read camera")
        break
    
    
    # converting image  to grayscale
    gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #detect faces 
    face = haar_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.3,
        minNeighbors=4
    ) 
    
    # draw rectangle around faces
    for (x,y,w,h) in face:
        cv2.rectangle(
            img,
            (x,y),        # x horixontal position, y vertical position, width, height
            (x+w,x+h),
            (255,255,0),
            3
        )
        
        
    # display camera
    cv2.imshow("Face Detection",img)
    
    # press esc to exit
    if cv2.waitKey(10) == 27:
        break
    
 # realease camera
cam.release()
    
# close windows
cv2.destroyAllWindows()