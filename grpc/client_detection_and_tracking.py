# USAGE
# python object_detection_and_tracking.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from collections import deque
import numpy as np
import time
import argparse
import imutils
import cv2
import sys
import random
import zmq;
import imageio
import psutil
import os
imageio.plugins.ffmpeg.download()

import skvideo.io
from moviepy.editor import VideoFileClip

from concurrent import futures
import grpc
import video_analytic_pb2
import video_analytic_pb2_grpc

import check_IOU

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

############################################################################
# Set up tracker.
# Instead of MIL, you can also use
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = tracker_types[0]

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
ap.add_argument("-i", "--input_video", required=True,
	help="path to the input video file")
ap.add_argument("-o", "--output_video", required=True,
	help="path to the output video file")
ap.add_argument("-d", "--directory_output", default=".",
	help="path to the output result of the algorithm")
ap.add_argument("-b", "--baseline", type=int, default=0,
	help="baseline=1, run the frame using the dnn algorithm")
ap.add_argument("-f", "--file_output", default="bbox_baseline.txt",
	help="name of output file")
ap.add_argument("-l", "--local", type=int, default=0,
	help="where to run the object detection, local=1, run local")
ap.add_argument("-s", "--server", default="node-1:50051",
	help="Sever of RPC default : node-1:50051")
ap.add_argument("-t", "--t_periodic", type=int, default=0,
	help="Periodicly send request to server")
ap.add_argument("-z", "--basefile", default="basefile.txt",
	help="The base line file")
ap.add_argument("-u", "--period", type=float, default=0.15,
	help="Periodicy, only if -t is specified")

args = vars(ap.parse_args())

totalTime = 0;
totalSendRecv = 0;


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

#####################################################################################################

def s_object_detection(stub, frame):
	global totalSendRecv
	global totalTime 

	totalSendRecv = totalSendRecv + 1;

	myframe = video_analytic_pb2.Image()
	myframe.height = frame.shape[0]
	#print myframe.height
	myframe.width = frame.shape[1]  
	#print myframe.width
	myframe.depth = frame.shape[2]
	#print myframe.depth
	myframe.dtype = str(frame.dtype)
	myframe.imagedata = np.ndarray.tobytes(frame)
	
	startTime = time.time();
	ObjDetectData = stub.DetectOneObject(myframe); 
	#ObjDetectData = stub.DetectOneObject(myframe); 
	
	endTime = time.time();
	totalTime = totalTime + (endTime - startTime);


	bbox =  ObjDetectData.bbox
	index =  ObjDetectData.index
	confidence = ObjDetectData.confidence


	label = "{}, {:.2f}%".format(CLASSES[int(index.value)],
				confidence.conf * 100)
	#print ("done array")
	return (True, (bbox.x, bbox.y, bbox.width, bbox.height), label)	

#####################################################################################################

def l_object_detection(frame):
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)

	(startX, startY, endX, endY) = (0,0,0,0)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	maxconfidence = -1;
	maxarea = -1;
	(sX, sY, eX, eY) = (0,0,0,0)
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(sX, sY, eX, eY) = box.astype("int")
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		area = (eY - sY) * (eX - sX)
		if confidence > args["confidence"]:
			if (area > maxarea):
				print detections.shape[2]
				print area;
				maxarea = area;
			else:
				continue;

			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
			label = "{}, {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	return (True, (startX, startY, endX-startX, endY-startY), label)		

####################################################################################################

def object_detection(stub, frame):
	if args["local"]:
		return l_object_detection(frame);
	else:
		return s_object_detection(stub, frame);


####################################################################################################

def framediff(prevframe, frame):
	# Change to gray scale
	prevframe_gryscale = cv2.cvtColor(prevframe, cv2.COLOR_BGR2GRAY)
	frame_gryscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Check the difference
	diff_frame = abs(frame_gryscale - prevframe_gryscale);


	condfunc = lambda t : (t > 35) and 1 or 0
	vfunc = np.vectorize(condfunc)
	# filter if > 35
	filter_frame  = vfunc(diff_frame);

	# sum value
	sum_frame = filter_frame.sum();
	
	# check the difference
	if sum_frame > 40000 :
		return False;
	else:
		return True;


#####################################################################################################
trackTimer = 0.0;
def frameprocessing(frame, prevframe, tracker, label, s, outputfile):
	global trackTimer;

	frame = imutils.resize(frame, width=400)

	# Start timer
	timer = cv2.getTickCount()
	
	# Not ok means frame is differnt diff
	#ok = framediff(prevframe, frame);
	ok = True
	bbox = (0,0,0,0)


	# Update tracker
	if ok:
		ok, bbox = object_tracking(frame, tracker);

	if not ok or args["baseline"] or (args["t_periodic"] and trackTimer >= args["period"]):
		ok, bbox, label = object_detection(s, frame);
		tracker.init(frame, bbox)
		prevframe = frame;

		if trackTimer >= args["period"]:
			trackTimer = 0;

	# Calculate Frames per second (FPS)
	fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

	trackTimer = trackTimer + 1/fps
	

	# Draw bounding box
	if ok:
		# Tracking success
		p1 = (int(bbox[0]), int(bbox[1]))
		p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
		cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

		outputfile.write(str(bbox[0]) + "," + str(bbox[1]) + "," + str(bbox[0] + bbox[2]) + "," + 
			str(bbox[1] + bbox[3]) + ",");

	else :
		# Tracking failure
		cv2.putText(frame, "Failure in object detection and tracking", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

	# Display FPS on frame
	cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
	# Display class label 
	startX = int ( bbox[0] );
	y = int (bbox[1] - 15) if bbox[1] - 15 > 15 else int(bbox[1] + 15)
	cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[0], 2)

	outputfile.write(label + "\n");

	# Display result
	return frame;


#####################################################################################################
# construct the argument parse and parse the arguments
if __name__ == "__main__":
	

	myProcess = psutil.Process(os.getpid())
	myProcess.cpu_percent();

	myStartTime = time.time();

	white_output = args["output_video"]
	clip1 = VideoFileClip(args["input_video"]);


	frame = clip1.get_frame(0);
	frame = imutils.resize(frame, width=400);

	prevframe = frame;
	
	# open a gRPC channel
	channel = grpc.insecure_channel(args["server"])
	stub = video_analytic_pb2_grpc.VideoAnalyticStub(channel)

	ok, bbox, label = object_detection(stub, frame);

	# Initialize tracker with first frame and bounding box
	ok = tracker.init(frame, bbox);

	# Initialize the output directory
	outputfilename = os.path.join(args["directory_output"], args["file_output"]);
	outputfile = open(outputfilename, "w");

	white_clip = clip1.fl_image(lambda x: frameprocessing(x, prevframe, tracker, label, stub, outputfile))
	white_clip.write_videofile(white_output, audio=False)
	outputfile.flush();
	outputfile.close();

	myEndTime = time.time();

	cpupercent = myProcess.cpu_percent()
	memorypercent = myProcess.memory_percent()


	sizedata = frame.shape[0]*frame.shape[1]*frame.shape[2]*4 + 6*4
	if totalTime != 0:
		print "Bandwidth : " + str( float(totalSendRecv * sizedata) / (totalTime * 1024*1024)) + " MByte/s"
	print "Cpu percent : " + str(cpupercent);
	print "Memory percent : " + str(memorypercent);
	print "Time : " + str(myEndTime - myStartTime);

	#if not args["baseline"]:
	print "Accuracy : " + str(check_IOU.calculate_IOU(args["basefile"], outputfilename));

	sys.exit();