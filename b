#Project Name: Computer Vision module application for finding a target in a live camera.

import cv2
# setting threshold 50 percent so any object below 50 percent is not recognised
thres = 0.5
classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath,configPath)
# set input frame size that is according to single shot detector file
net.setInputSize(320,320)
#  It can reduce the time of training of a neural network as more is the number of pixels in an image more is the number of input nodes that in turn increases the complexity of the model.
# Set scalefactor value for frame.
net.setInputScale(1.0/ 127.5)
# Set mean value for frame.
# sum of all frame /no. of frame 
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
def getObjects(img, thresh, nms, draw = True,objects = []):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #Non Maximum Suppression
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId-1]
            # for x in range(count):
            #count = x + count
            #print("hello",count[x])
            if className in objects:
                objectInfo.append([className])
                print(len(objectInfo))
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,className.upper(),(box[0]+10,box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    return img, objectInfo

if __name__ == "__main__":
    #Here it would help if you defined your video path in the parameter.
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    #cap.set(10,70)
    while True:
        success,img = cap.read()
        result, objectInfo = getObjects(img, 0.45, 0.2, objects = [])
        print(objectInfo)
        cv2.imshow("Output",img)
        # press q to exit the window
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()