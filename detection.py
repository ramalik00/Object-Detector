import numpy as np
import cv2

Min_Conf=0.3
NMS_THRESH=0.3
def detect_objects(frame, net,last_layer):
	(H, W) = frame.shape[:2]
	results = []
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(last_layer)
	boxes = []
	confidences = []
	classes=[]

	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > Min_Conf:
                                
                                box = detection[0:4] * np.array([W, H, W, H])
                                (X_center,Y_center,w,h)=box.astype("int")
                                x=int(X_center-(w/2))
                                y=int(Y_center-(h/2))
                                boxes.append([x, y, int(w), int(h)])
                                
                                confidences.append(float(confidence))
                                classes.append(classID)
				

	indexs = cv2.dnn.NMSBoxes(boxes, confidences, Min_Conf, NMS_THRESH)

	
	if len(indexs) > 0:
		
		for i in indexs.flatten():
			(x,y) = (boxes[i][0], boxes[i][1])
			(w,h) = (boxes[i][2], boxes[i][3])
			r = (confidences[i], (x, y, x + w, y + h),classes[i])
			results.append(r)

	
	return results
