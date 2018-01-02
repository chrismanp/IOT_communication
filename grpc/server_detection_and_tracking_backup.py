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