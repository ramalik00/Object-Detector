

from detection import detect_people
import numpy as np
import argparse
import cv2
import imutils
import os


USE_GPU=False
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="0",)
ap.add_argument("-o", "--output", type=str, default="")
ap.add_argument("-d", "--display", type=int, default=1,)
args = vars(ap.parse_args())


labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])
print("MODEL LOADED")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


if USE_GPU:
	
	print("setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


last_layer = net.getLayerNames()
last_layer = [last_layer[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print("INPUT LOADED")
vs = cv2.VideoCapture(args["input"])
writer = None

while True:
	
	(access, frame) = vs.read()
	if not access:
		break
	
	frame = imutils.resize(frame, width=700)
	results = detect_people(frame, net, last_layer)

	for (i, (prob, bounding_box, centroid,classes)) in enumerate(results):
		
		(X_start, Y_start, X_end, Y_end) = bounding_box
		(X_c, Y_c) = centroid
		cv2.rectangle(frame, (X_start, Y_start), (X_end, Y_end), (0,255,0), 2)
		text=LABELS[classes]
		cv2.putText(frame,text,(X_start,Y_start-10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
		text = "Total Objects"+str(len(results))
		cv2.putText(frame, text, (10, frame.shape[0] - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

	if args["display"] > 0:
		
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		if key == ord("q"):
			break

	if args["output"] != "" and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)

	if writer is not None:
		writer.write(frame)
