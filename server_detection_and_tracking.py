# USAGE
# python object_detection_and_tracking.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import sys
import random
import zmq;
import imageio
imageio.plugins.ffmpeg.download()

import skvideo.io
from moviepy.editor import VideoFileClip

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

############################################################################
# Set up tracker.
# Instead of MIL, you can also use
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = tracker_types[4]

if int(minor_ver) < 3:
	tracker = cv2.Tracker_create(tracker_type)
else:
	if tracker_type == 'BOOSTING':
		tracker = cv2.TrackerBoosting_create()
	if tracker_type == 'MIL':
		tracker = cv2.TrackerMIL_create()
	if tracker_type == 'KCF':
		tracker = cv2.TrackerKCF_create()
	if tracker_type == 'TLD':
		tracker = cv2.TrackerTLD_create()
	if tracker_type == 'MEDIANFLOW':
		tracker = cv2.TrackerMedianFlow_create()
	if tracker_type == 'GOTURN':
		tracker = cv2.TrackerGOTURN_create()

####################################################################

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
	help="max buffer size")
args = vars(ap.parse_args())

#####################################################################

def send_array(socket, A, flags=0, copy=True, track=False):
	"""send a numpy array with metadata"""
	md = dict(
		dtype = str(A.dtype),
		shape = A.shape,
	)
	socket.send_json(md, flags|zmq.SNDMORE)
	return socket.send(A, flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
	"""recv a numpy array"""
	md   = socket.recv_json(flags=flags)
	msg  = socket.recv(flags=flags, copy=copy, track=track)
	buf  = buffer(msg)
	A    = np.frombuffer(buf, dtype=md['dtype'])
	return A.reshape(md['shape'])

#####################################################################
def object_tracking(frame, tracker):
	ok, bbox = tracker.update(frame)
	return (ok, bbox);

#############################################################################################################
# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

def object_detection(frame):
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)

	(startX, startY, endX, endY) = (0,0,0,0)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in np.arange(0,1): #np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	return (True, (startX, startY, endX-startX, endY-startY), idx, confidence)	

def s_object_detection(s, frame):
	send_array(s, frame)
	bbox_label = 	recv_array(s)
	label = "{}: {:.2f}%".format(CLASSES[bbox_label[4]],
				confidence * 100)
	return (True, (bbox_label[0], bbox_label[1], bbox_label[2], bbox_label[3]), label)	

####################################################################################################

def frameprocessing(frame, tracker, label, s):
	frame = imutils.resize(frame, width=400)

	# Start timer
	timer = cv2.getTickCount()
	# Update tracker
	ok, bbox = object_tracking(frame, tracker);

	if not ok :
		ok, bbox, label = s_object_detection(s, frame);
		ok = tracker.init(frame, bbox)

	# Calculate Frames per second (FPS)
	fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
	# Draw bounding box
	if ok:
		# Tracking success
		p1 = (int(bbox[0]), int(bbox[1]))
		p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
		cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
	else :
		# Tracking failure
		cv2.putText(frame, "Failure in object detection and tracking", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

	# Display FPS on frame
	cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
	# Display class label 
	startX = int ( bbox[0] );
	y = int (bbox[1] - 15) if bbox[1] - 15 > 15 else int(bbox[1] + 15)
	cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[0], 2)

	# Display result
	return frame;


#####################################################################################################
# construct the argument parse and parse the arguments
if __name__ == "__main__":
	ctx = zmq.Context.instance();
	s = ctx.socket(zmq.REP)
	addr = 'tcp://127.0.0.1:9999'
	s.bind(addr)

	while True:
		print ("recv array")	
		frame = recv_array(s);
		ok, bbox, label, confidence = object_detection(frame)
		result = np.array([bbox[0],bbox[1],bbox[2],bbox[3], label, confidence])
		print ("send array")
		send_array(s, result)

	s.disconnect(addr)
	s.close();
	sys.exit();