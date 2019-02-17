#Nicholas Barry James Eginton
#2019-VisionCode
import io
import cv2
import numpy as np
import os
import pickle
#from networktables import NetworkTable
from time import sleep
import time
import math
import socket
from threading import Thread
import cv2


def twos_comp(val, num_bits):
    val = int(val)
    if val < 0:
        val = 256 + val
    return val


###########################################################################
# Constants
###########################################################################
CAMERA_OFFSET    = 12 # DISTANCE OF CMAERA FROM CENTER OF ROBOT
YPOSITION        = 48 # HEIGHT OF CAMERA
ZPOSITION        = 14 # DISTANCE FROM FRONT OF BOT
RECTANGLE_RATIO  = 3.3134/5.8256 #Approximate value of the height and width of target boundingRect
FOV_PERPIXEL     = 0.064 
A_TO_R           = math.pi/180
FOV_PERPIXEL_RAD = FOV_PERPIXEL * A_TO_R
CAMERA_ANGLE     = 0 # Camera angle to robot in radians
UDP_IP = "10.6.39.2" #UDP IP
UDP_PORT = 5810 #UDP PORT
#This creates a new socket to send data to
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
UDP_ADDRESS      = (UDP_IP, UDP_PORT)
###########################################################################
# This class starts a thread that continuously reads a frame from a camera
# and makes it available to the main program
###########################################################################
class WebcamVideoStream:
	def __init__(self, src=0):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()

		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False

	def start(self):
		# start the thread to read frames from the video stream
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return

			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
		# return the frame most recently read
		return self.frame

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
		
##########################################################################
# This function is called whenever the user changes the trackbars.
# It grabs the current trackbar settings and places them into FilterValues
# It stores the values to saveFilter.p using pickle
##########################################################################
def SaveParameters(*arg):
    # See if the exposure changed. If so, write it to the camera.
    if (FilterValues["EXPOS"] != cv2.getTrackbarPos("EXPOS", "Tracker")):
        os.system('sudo v4l2-ctl -c exposure_auto=1 -c exposure_absolute='+str(cv2.getTrackbarPos("EXPOS", "Tracker")))

    FilterValues["H_MIN"] = cv2.getTrackbarPos("H_Min", "Tracker")
    FilterValues["H_MAX"] = cv2.getTrackbarPos("H_Max", "Tracker")
    FilterValues["S_MIN"] = cv2.getTrackbarPos("S_Min", "Tracker")
    FilterValues["S_MAX"] = cv2.getTrackbarPos("S_Max", "Tracker")
    FilterValues["V_MIN"] = cv2.getTrackbarPos("V_Min", "Tracker")
    FilterValues["V_MAX"] = cv2.getTrackbarPos("V_Max", "Tracker")
    FilterValues["EXPOS"] = cv2.getTrackbarPos("EXPOS", "Tracker")
    output = open("saveFilter.p", "wb")
    pickle.dump(FilterValues,output)
    output.close
    
#####################################################################3
def greaterArea(a, b):
    if cv2.contourArea(a) > cv2.contourArea(b):
        return -1
    return 1
   
#####################################################################3
def ExitCallback():
    exit
    
def GetCenter(box):
    return ((box[0][0] + box[1][0] + box[2][0] + box[3][0]) // 4, (box[0][1] + box[1][1] + box[2][1] + box[3][1]) // 4)
                        
#######################################
# This is the start of the main program
#######################################
cam = WebcamVideoStream(src=0).start()
#cam2 = WebcamVideoStream(src=1).start()


img=cam.read() # Throw away the first frame
while img is None:
    img = cam.read()
    time.sleep(0.2)

# Create the two named windows
cv2.namedWindow("Tracker",cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("camera",cv2.WINDOW_AUTOSIZE)

# cameraSource is used to switch between different views.
# If it is 1 then show the camera image with the results of
#   of the tracking added to the image.
# If it is 2 then show the results of the filtering.
cameraWindowSource = 1


# Set up a dictionary to store the filter parameters. Assign initial values.
FilterValues = {"H_MIN":52,"H_MAX":86,"S_MIN":77,"S_MAX":255,"V_MIN":111,"V_MAX":255,"EXPOS":200}

# Try to load the previous values from a pickle file.
try:
    FilterValues = pickle.load(open("saveFilter.p", "rb"))
except IOError:
    print ("No parameter file")

cv2.createTrackbar("H_Min", "Tracker", FilterValues["H_MIN"], 255, SaveParameters)
cv2.createTrackbar("H_Max", "Tracker", FilterValues["H_MAX"], 255, SaveParameters)
cv2.createTrackbar("S_Min", "Tracker", FilterValues["S_MIN"], 255, SaveParameters)
cv2.createTrackbar("S_Max", "Tracker", FilterValues["S_MAX"], 255, SaveParameters)
cv2.createTrackbar("V_Min", "Tracker", FilterValues["V_MIN"], 255, SaveParameters)
cv2.createTrackbar("V_Max", "Tracker", FilterValues["V_MAX"], 255, SaveParameters)
cv2.createTrackbar("EXPOS", "Tracker", FilterValues["EXPOS"], 500, SaveParameters)

# This will attempt to set the exposure of the camera.
# It is not very reliable.
os.system('sudo v4l2-ctl -d=2 -c exposure_auto=1 -c exposure_absolute='+str(FilterValues["EXPOS"]))

# set up the network table as a server
#sd = NetworkTable.getTable("CameraTracker")
# set up the network table as a server
#   NetworkTable.initialize(server="roborio-639-FRC.local")
#NetworkTable.setIPAddress("roborio-639-FRC.local")
#   sd = NetworkTable.getTable("CameraTracker")	


XAngleToTarget = 45

TargetTime = time.time()
start_timer = time.time()

# The main loop.  Pressing the esc key will exit the loop and stop the program
while(True):
   # print (time.time() - start_timer)  #Used to measure cycle time

    start_timer = time.time()
    
    CameraDistance = -1 # This is the value we report if we don't find a target
    
    img=cam.read() # Grab an image
    #print(img)
    #img2 = cam2.read()
#    cv2.imshow("camera2",img2)

#        self.rgb, self.frame = f, cv2.cvtColor(f,cv2.COLOR_BGR2HSV) # Convet it to HSV format

    HSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) # Convert it to HSV format
    # Filter the HSV image to keep only the color range specified by the Filter values.
    mask = cv2.inRange(HSV, np.array([FilterValues["H_MIN"],FilterValues["S_MIN"],FilterValues["V_MIN"]]), np.array([FilterValues["H_MAX"],FilterValues["S_MAX"],FilterValues["V_MAX"]]))
    mask = cv2.erode(mask, None, iterations=2) # These two functions get rid of noise
    mask = cv2.dilate(mask, None, iterations=2)
    
    
    if cameraWindowSource == 2:
        cv2.imshow("camera",mask) # Display the filter results

    # find contours in the mask and initialize the current
    # (x, y) center of the object
    allContours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # Draw the center line
    cv2.line(img,(320,1), (320,479), (0,0,0), 1)
    # for 320x240 cv2.line(img,(160,1), (160,240), (0,0,0), 2)

       
    # Draw the instructional text
    cv2.putText(img,"press f for filter, n for normal, Esc to quit",(20,20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)

# James and Nick's Code Part I 

#Gets values from camers for processing   
    minAreaRects    = []
    minAreaCorners  = []
#    boundingCorners = []    
    for cont in allContours:
        
        tempMinAreaRect = cv2.minAreaRect(cont) # creates temp minAreaRect of the contours from the output
        tempBox = cv2.boxPoints(tempMinAreaRect) # creates tuple of the corners of tempMinAreaRects
        tempBox = np.int0(tempBox) # converts all points to integers
        tempBoundingRect = cv2.boundingRect(cont) # creates temp boundingRect of the contours from the output
        # print(tempBoundingRect)
        
        #Find out which contours are targets, by seeing if they have the correct aspect ratio and if they are tall enough to be in range
        #For tempBoundingRect[0]-X [1]-Y [2]- Width [3]- Height
        CurrentTargetRatio = tempBoundingRect[2] / tempBoundingRect[3]
        
       
        if CurrentTargetRatio > RECTANGLE_RATIO -.2 \
           and CurrentTargetRatio < RECTANGLE_RATIO +.2\
           and tempBoundingRect[3] > 50:

            cv2.drawContours(img, [tempBox],0,(0,0,255),1) # Draw the rotated box on the imagev    

            minAreaCorners.append(tempBox) #Creates a tuple of minAreaRect corners
    
    # Continue if there is more than one targets    
    if len(minAreaCorners) > 1:
        # Find target closest to the center    
        closestToCenter = 0
        # Get the first target
        previousCenter = GetCenter(minAreaCorners[0])
        cv2.circle(img,(int(previousCenter[0]),int(previousCenter[1])),2,(0,0,255),-1) #Draws a circle at the centers of the targets
        
        # Step through and check the remaining targets to see if any are closer to the center
        for i in range(1, len(minAreaCorners)):
            center = GetCenter(minAreaCorners[i])
            if abs(center[0] - 320) < abs(previousCenter[0]-320): 
                closestToCenter = i             
                previousCenter = center
            cv2.circle(img,(int(center[0]),int(center[1])),2,(0,0,255),-1) #Draws a circle at the centers of the targets

        #closestToCenter is the index into minAreaCorners that points to the target closest to center
        
        # in the centermost target, find the rightmost point to determine if the target slopes left or right
        rightMostPoint = minAreaCorners[closestToCenter][0] #Set rightmost point
   
        for i in range(0, 4):
            #Within the center target the right most point and assign that value to it
            if (minAreaCorners[closestToCenter][i][0] > rightMostPoint[0]):
                rightMostPoint = minAreaCorners[closestToCenter][i]
 
        # This is just for display purposes
        if rightMostPoint[1] > previousCenter[1]:        
            cv2.circle(img,(int(previousCenter[0]),int(previousCenter[1])),30,(255,0,0),0)
        else:
            cv2.circle(img,(int(previousCenter[0]),int(previousCenter[1])),30,(0,255,0),0)
       
        # Find the second target
        SecondTarget = None
        for i in range(0, len(minAreaCorners)):

            # Determine if the centermost target is angled left or right        
            if rightMostPoint[1] > previousCenter[1]: 
                if GetCenter(minAreaCorners[i])[0] < previousCenter[0]:
                    # This target is to the left of the center target     
                    if SecondTarget is None:
                        SecondTarget = minAreaCorners[i]
                    else:
                        if SecondTarget[0][0] < minAreaCorners[i][0][0]:
                            SecondTarget = minAreaCorners[i]
            else:
                if GetCenter(minAreaCorners[i])[0] > previousCenter[0]:
                    # This target is to the right of the center target     
                    if SecondTarget is None:
                        SecondTarget = minAreaCorners[i]
                    else:
                        if SecondTarget[0][0] > minAreaCorners[i][0][0]:
                            SecondTarget = minAreaCorners[i]
            

        if SecondTarget is not None:
             cv2.line(img,GetCenter(SecondTarget),previousCenter,(255,0,0),2) # Draw a line from the closest to center target to is found to be respective target
             centerOfLine = ((SecondTarget[0][0] + SecondTarget[1][0])/2, (SecondTarget[0][1] + SecondTarget[1][1])/2)
             length = abs(GetCenter(SecondTarget)[0] - previousCenter[0])
             distance = (11.3134)/math.tan(length*FOV_PERPIXEL_RAD)
             angleToTarget = (((GetCenter(SecondTarget)[0] + previousCenter[0])/2)-320)*FOV_PERPIXEL
     #        distance = distance * math.cos(CAM_ANGLE) - ZPOSITION
             if distance < 72:
                 CameraDistance = distance
             if angleToTarget > -20 and angleToTarget < 20:
                 CameraAngle = (math.atan((distance * math.sin(angleToTarget*A_TO_R + CAMERA_ANGLE) - CAMERA_OFFSET) / distance))/A_TO_R
             imageText ="Angle:"+str(int(CameraAngle * 10)/10.0) +" Distance:"+str(int(distance *10)/10.0)
             cv2.putText(img,imageText,(5,470), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
             print("d:{},a:{}".format(CameraDistance, CameraAngle))
             sock.sendto(bytes([twos_comp(CameraDistance, 8), twos_comp(CameraAngle * 5, 8)]), UDP_ADDRESS)
        else:
             imageText ="No Target"
             sock.sendto(bytes([twos_comp(-1, 8),twos_comp(-69, 8)]), UDP_ADDRESS)
             cv2.putText(img,imageText,(5,470), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
             print(imageText)         
    else:
        avgCount = 0
        avgX = 0
    if cameraWindowSource == 1:
        cv2.imshow("camera",img)


        
    k = cv2.waitKey(10)
        
    if k == 27:
        break # Exit the program
    
    if k == 102: # f key - display the results of the filter
        cameraWindowSource = 2
    if k == 110: # n key - display the camera image with the results added
        cameraWindowSource = 1
    if k == 103: # g key
        cameraWindowSource = 3
        
print("Exiting")
#cam.release()
cam.stop()
#cam2.stop()
# There is an issue closing the windows.
# Stackoverflow suggested the following code
for i in range(1,10):
    cv2.destroyAllWindows()
    cv2.waitKey(1)



