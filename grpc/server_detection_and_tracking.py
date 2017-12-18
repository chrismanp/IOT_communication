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
import skvideo.io
from moviepy.editor import VideoFileClip

from concurrent import futures
import grpc
import video_analytic_pb2
import video_analytic_pb2_grpc

import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

#####################################################################
# What model to download.
MODEL_NAME = 'faster_rcnn_resnet50_coco_2017_11_08'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

NUM_CLASSES = 90

#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
	file_name = os.path.basename(file.name)
	if 'frozen_inference_graph.pb' in file_name:
		tar_file.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')


####################################################################
if tf.__version__ != '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
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
			label = "{}, {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	return (True, (startX, startY, endX-startX, endY-startY), idx, confidence)	

def object_detection_tensorflow(frame):
	with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as sess:
			# Definite input and output Tensors for detection_graph
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			# Each box represents a part of the image where a particular object was detected.
			detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			# Each score represent how level of confidence for each of the objects.
			# Score is shown on the result image, together with the class label.
			detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
			detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')

			image_np_expanded = np.expand_dims(frame, axis=0)
			# Actual detection.
			(boxes, scores, classes, num) = sess.run(
				[detection_boxes, detection_scores, detection_classes, num_detections],
				feed_dict={image_tensor: image_np_expanded})

			max_confidence = -1;
			max_index = -1;
			max_class = -1;
			for i in np.arange(0,int(num)): #np.arange(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated with
				# the prediction
				confidence = scores[0][i]

				# filter out weak detections by ensuring the `confidence` is
				# greater than the minimum confidence
				if confidence > args["confidence"]:
					if (confidence > max_confidence):
						max_confidence = confidence
						max_class = classes[0][i]
						max_index = i;
					else:
						continue;
					# extract the index of the class label from the
					# `detections`, then compute the (x, y)-coordinates of
					# the bounding box for the object
	print "Frame shape : "			
	print frame.shape		
	startX = int(boxes[0][max_index][1] *  frame.shape[1])
	startY = int(boxes[0][max_index][0] *  frame.shape[0])
	endX   = int(boxes[0][max_index][3] *  frame.shape[1]) 
	endY   = int(boxes[0][max_index][2] *  frame.shape[0])
	
	return (True, (startX, startY, endX-startX, endY-startY), int(max_class), max_confidence)


class VideoAnalyticServicer(video_analytic_pb2_grpc.VideoAnalyticServicer):
	"""Provides methods that implement functionality of VideoAnalticServicer."""

	def DetectOneObject(self, framecontext, context):
		frame = np.frombuffer(framecontext.imagedata, dtype=framecontext.dtype)
		frame = frame.reshape((framecontext.height, framecontext.width, framecontext.depth))

		ok, bbox, label, confidence = object_detection(frame);

		myObjDetectData = video_analytic_pb2.ObjDetectData()

		myObjDetectData.confidence.conf = confidence
		myObjDetectData.index.value = label
		myObjDetectData.bbox.x = bbox[0]
		myObjDetectData.bbox.y = bbox[1]
		myObjDetectData.bbox.width=bbox[2]
		myObjDetectData.bbox.height =bbox[3]

		return myObjDetectData

	def DetectOneObject_tensorflow(self, framecontext, context):
		frame = np.frombuffer(framecontext.imagedata, dtype=framecontext.dtype)
		frame = frame.reshape((framecontext.height, framecontext.width, framecontext.depth))

		ok, bbox, label, confidence = object_detection_tensorflow(frame);

		print ok
		print bbox
		print label
		print confidence

		myObjDetectData = video_analytic_pb2.ObjDetectData()

		myObjDetectData.confidence.conf = confidence
		myObjDetectData.index.value = label
		myObjDetectData.bbox.x = bbox[0]
		myObjDetectData.bbox.y = bbox[1]
		myObjDetectData.bbox.width=bbox[2]
		myObjDetectData.bbox.height =bbox[3]

		return myObjDetectData


#####################################################################################################
# construct the argument parse and parse the arguments
if __name__ == "__main__":
	server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
	video_analytic_pb2_grpc.add_VideoAnalyticServicer_to_server(
	  VideoAnalyticServicer(), server)
	server.add_insecure_port('[::]:50051')
	server.start()

	print ("Server Start")
	try:
		while True:
			time.sleep(_ONE_DAY_IN_SECONDS)
	except KeyboardInterrupt:
		server.stop(0)