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

import time
import cv2
import skvideo.io
from moviepy.editor import VideoFileClip

import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile


(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

############################################################################
# Set up tracker.
# Instead of MIL, you can also use
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = tracker_types[2]

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
ap.add_argument("-s", "--server",default="10.1.1.3:9999",
	help="location of the server")
ap.add_argument("-a", "--areathreshold", type=int, default=2800,
	help="Area threshold")
args = vars(ap.parse_args())

#####################################################################
# Setup the tensor flow integration

# What model to download.
#MODEL_NAME = 'faster_rcnn_resnet50_coco_2017_11_08'
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
#MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
#MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2017_11_08'
MODEL_NAME = 'faster_rcnn_resnet50_lowproposals_coco_2017_11_08'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

NUM_CLASSES = 90

# Download model if not downloaded
if not os.path.exists(PATH_TO_CKPT):
	print "Download model " +  DOWNLOAD_BASE + MODEL_FILE
	opener = urllib.request.URLopener()
	opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
	tar_file = tarfile.open(MODEL_FILE)
	for file in tar_file.getmembers():
		file_name = os.path.basename(file.name)
		if 'frozen_inference_graph.pb' in file_name:
	 		tar_file.extract(file, os.getcwd())

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

# Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Object detection
def object_detection_tensorflow(frame):
	list_of_obj = [];

	starttimer = cv2.getTickCount()
	stime = time.time();


	with detection_graph.as_default():
		with tf.Session(graph = detection_graph) as sess:
			# Definite input and output Tensors for detection_graph
			image_tensor      = detection_graph.get_tensor_by_name('image_tensor:0')
			# Each box represents a part of the image where a particular object was detected.
			detection_boxes   = detection_graph.get_tensor_by_name('detection_boxes:0')
			# Each score represent how level of confidence for each of the objects.
			# Score is shown on the result image, together with the class label.
			detection_scores  = detection_graph.get_tensor_by_name('detection_scores:0')
			detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections    = detection_graph.get_tensor_by_name('num_detections:0')

			image_np_expanded = np.expand_dims(frame, axis=0)
			# Actual detection.
			stime2 = time.time();
			(boxes, scores, classes, num) = sess.run(
				[detection_boxes, detection_scores, detection_classes, num_detections],
				feed_dict={image_tensor: image_np_expanded})
			processingtimeusingtime2 = time.time() - stime2
			
			max_confidence = -1;
			max_index = -1;
			max_class = -1;
			for i in np.arange(0,int(num)): #np.arange(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated with
				# the prediction
				confidence = scores[0][i]
				#print "Frame shape : "			
				#print frame.shape		
				# filter out weak detections by ensuring the `confidence` is
				# greater than the minimum confidence
				if confidence > args["confidence"]:
					startX = float(boxes[0][i][1] *  frame.shape[1])
					startY = float(boxes[0][i][0] *  frame.shape[0])
					endX   = float(boxes[0][i][3] *  frame.shape[1]) 
					endY   = float(boxes[0][i][2] *  frame.shape[0])
					

					# Store the list object
					idx = classes[0][i]
					myarea = (endY - startY) * (endX - startX)					
					if(myarea >= args["areathreshold"]):
						list_of_obj.append([float(startX), float(startY), float(endX-startX), float(endY-startY), float(idx), float(confidence)])

											
	processingtimeusingtime = time.time() - stime
	processingtime = (cv2.getTickCount()- starttimer)/cv2.getTickFrequency()
	print "Processing Time : " + str(processingtimeusingtime) ;
	print "Processing Time 2 : " + str(processingtimeusingtime2) ;
	return list_of_obj;


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

CLASSES_2 = ["empty","person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


# load our serialized model from disk
#print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

def object_detection(frame):


	starttimer = cv2.getTickCount()
	stime = time.time();

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	#blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)
	blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)


	(startX, startY, endX, endY) = (0,0,0,0)

	# pass the blob through the network and obtain the detections and
	# predictions
	
	net.setInput(blob)
	detections = net.forward()

	list_of_obj = []

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# 'detections', then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			#box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			box = detections[0, 0, i, 3:7] * np.array([640, 360, 640, 360])
			(startX, startY, endX, endY) = box.astype("int")

			myarea = (endY - startY) * (endX - startX)

			# draw the prediction on the frame
			#label = "{}: {:.2f}%".format(CLASSES[idx],
			#	confidence * 100)
			#cv2.rectangle(frame, (startX, startY), (endX, endY),
			#	COLORS[idx], 2)
			#y = startY - 15 if startY - 15 > 15 else startY + 15
			#cv2.putText(frame, label, (startX, y),
			#	cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			
			if(myarea >= args["areathreshold"]):
				list_of_obj.append([float(startX), float(startY), float(endX-startX), float(endY-startY), float(idx), float(confidence)])

	processingtimeusingtime = time.time() - stime
	processingtime = (cv2.getTickCount()- starttimer)/cv2.getTickFrequency()
	print (processingtimeusingtime) ;
	return list_of_obj;

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
	addr = 'tcp://' + args["server"] #'tcp://10.1.1.3:9999'
	s.bind(addr)

	print "Ready : "
	while True:
		#print ("recv array")	
		frame = recv_array(s);
		#list_of_obj = object_detection(frame)
		list_of_obj = object_detection_tensorflow(frame)
		
		result = np.array(list_of_obj)
		#print ("send array")
		send_array(s, result)

	s.disconnect(addr)
	s.close();
	sys.exit();