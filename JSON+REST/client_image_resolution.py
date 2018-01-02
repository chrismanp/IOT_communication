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
import copy
imageio.plugins.ffmpeg.download()

import skvideo.io
from moviepy.editor import VideoFileClip

from concurrent import futures
import grpc
#import video_analytic_pb2
#import video_analytic_pb2_grpc

import check_IOU

SCRATCH_DIR = "/home/mnt_sdc/scratch_data/"

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

############################################################################
# Set up tracker.
# Instead of MIL, you can also use
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = tracker_types[2]

tracker = []
for i in range(0,10):
	if int(minor_ver) < 3:
		tracker.append ( cv2.Tracker_create(tracker_type) )
	else:
		if tracker_type == 'BOOSTING':
			tracker.append ( cv2.TrackerBoosting_create() )
		if tracker_type == 'MIL':
			tracker.append ( cv2.TrackerMIL_create() )
		if tracker_type == 'KCF':
			tracker.append ( cv2.TrackerKCF_create() )
		if tracker_type == 'TLD':
			tracker.append ( cv2.TrackerTLD_create() )
		if tracker_type == 'MEDIANFLOW':
			tracker.append ( cv2.TrackerMedianFlow_create() )
		if tracker_type == 'GOTURN':
			tracker.append ( cv2.TrackerGOTURN_create() )

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
ap.add_argument("-s", "--server", default="10.1.1.3:9999",
	help="Sever of RPC default : 10.1.1.3:9999")
ap.add_argument("-t", "--t_periodic", type=int, default=0,
	help="Periodicly send request to server")
ap.add_argument("-z", "--basefile", default="basefile.txt",
	help="The base line file")
ap.add_argument("-u", "--period", type=float, default=1,
	help="Periodicy, only if -t is specified")
ap.add_argument("-r", "--readfrommemory", type=int, default=0,
	help="Turn on to get data from file instead of running algorithm");
ap.add_argument("-x", "--pixeltheshold", type=int, default=50,
	help="Pixel wise threshold");
ap.add_argument("-k", "--sumtheshold", type=int, default=35000,
	help="Sum wise threshold");
ap.add_argument("-y", "--framediffdetector", type=int, default=0,
	help="1 = use frame diff detector, other wise not (default:0)");
ap.add_argument("-w", "--width", type=int, default=300,
	help="The width of frame sent to server");
ap.add_argument("-n", "--height", type=int, default=300,
	help="The height of frame sent to server");
ap.add_argument("-e", "--intermediate_width", type=int, default=400,
	help="The intermediate width ");
ap.add_argument("-v", "--resolutionretried", type=int, default=1,
	help="1=retry sending higher resolution");
args = vars(ap.parse_args())

totalTime = 0;
totalSendRecv = 0;

#####################################################################

clienserverttotalBytes = 0;
serverclienttotalBytes = 0;



def send_array(socket, A, flags=0, copy=True, track=False):
	"""send a numpy array with metadata"""
	
	global clienserverttotalBytes;
	md = dict(
		dtype = str(A.dtype),
		shape = A.shape,
	)
	socket.send_json(md, flags|zmq.SNDMORE)
	print "send Image shape : "
	print A.shape
	print "recv Image type : " + str(A.dtype)

	print  A.shape[0] * A.shape[1]* A.shape[2] * 1;
	clienserverttotalBytes = clienserverttotalBytes +  A.shape[0] * A.shape[1]* A.shape[2] * 1

	return socket.send(A, flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
	"""recv a numpy array"""
	
	global serverclienttotalBytes;

	md   = socket.recv_json(flags=flags)
	msg  = socket.recv(flags=flags, copy=copy, track=track)
	buf  = buffer(msg)
	print "send Image shape : "

	if md['shape'] != [0]:
		print md['shape']
		print "recv Image type : " + str(md['dtype'])	

		print md['shape'][0] * md['shape'][1] * 8
		serverclienttotalBytes = serverclienttotalBytes +  md['shape'][0] * md['shape'][1] * 8


		A    = np.frombuffer(buf, dtype=md['dtype'])
		return A.reshape(md['shape'])
	else:
		return [0];

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

def l_object_detection(frame):
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)

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

			list_of_obj.append([startX, startY, endX-startX, endY-startY, idx, confidence])

	return np.array(list_of_obj);

#####################################################################################################
def s_object_detection(s, frame):
	#print ("send array")
	global totalSendRecv
	global totalTime 
	
	totalSendRecv = totalSendRecv + 1;
	startTime = time.time();
	
	
	send_array(s, frame)
	#print ("recv array")
	bbox_label = 	recv_array(s)
	endTime = time.time();
	totalTime = totalTime + (endTime - startTime);

	#print ("done array")
	return np.array(bbox_label);	

####################################################################################################

def framediff(prevframe, frame, numberofframes, prevframe_index_global):
	scale=255;

	# Change to gray scale

	prevframe_gryscale = cv2.cvtColor(prevframe, cv2.COLOR_BGR2GRAY)
	frame_gryscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	
	# Denoise frame
	#prevframe_gryscale = cv2.fastNlMeansDenoising(prevframe_gryscale,None,3)
	#frame_gryscale = cv2.fastNlMeansDenoising(frame_gryscale,None,3)

	# Equalize histogram
	#prevframe_gryscale = cv2.equalizeHist(prevframe_gryscale)
	#frame_gryscale = cv2.equalizeHist(frame_gryscale)


	#imagename = SCRATCH_DIR + args["directory_output"] + "image_dir/gray_scale_image_first" +str(args["period"]) + str(numberofframes)  + "_.bmp"
	#cv2.imwrite(imagename, frame);

	#imagename = SCRATCH_DIR + args["directory_output"] + "image_dir/gray_scale_image_prev" +str(args["period"]) + str(prevframe_index_global)  + "_.bmp"
	#cv2.imwrite(imagename, prevframe);

	# Check the difference
	diff_frame = abs(frame_gryscale - prevframe_gryscale);

	#imagename = SCRATCH_DIR + args["directory_output"] + "image_dir/gray_scale_imagediff" +str(args["period"]) + str(numberofframes) + "-" + str(prevframe_index_global)  + "_.bmp"
	#cv2.imwrite(imagename, diff_frame);

	condfunc = lambda t : (t > 75 and t < 180) and scale or 0
	vfunc = np.vectorize(condfunc)
	# filter if > 35
	filter_frame  = vfunc(diff_frame);
	#cv2.imshow("image",frame_gryscale);
	#k = cv2.waitKey(0)
	# sum value
	
	imagename = SCRATCH_DIR + args["directory_output"] + "image_dir/filter_frame_imagediff" +str(args["period"]) + str(numberofframes) + "-" + str(prevframe_index_global)  + "_.bmp"
	cv2.imwrite(imagename, filter_frame);


	sum_frame = filter_frame.sum();
	
	# check the difference
	if sum_frame > args["sumtheshold"] * scale :
		print "Difference : "
		print filter_frame
		print sum_frame;
		return True;
	else:
		return False;


#####################################################################################################
trackFrame = 0;
ntracker = 0;
trackingTimes = 0.0;
independentTrackingTimes = 0;
numberoftracking = 0;
objdetectionTimes = 0.0;
numberofobjdetection = 0;
numberofframes = 0;
perframeTimes = 0;
cpupercent_total = 0.0;
memorypercent_total = 0.0;
confidence_arr_global = [];
index_arr_global = [];
prevframe_index_global =1;
prevframe_global = np.array([]);
frame_index = 0;
retried_res = 0;
def frameprocessing(frame, label, local_tracker, s, outputfile, bbox_data):
	global ntracker;
	global trackingTimes ;
	global objdetectionTimes ;
	global numberoftracking;
	global numberofobjdetection;
	global independentTrackingTimes;
	global numberofframes ;
	global perframeTimes;
	global cpupercent_total;
	global memorypercent_total;
	global confidence_arr_global;
	global index_arr_global;
	global prevframe_global;
	global prevframe_index_global;
	global frame_index;
	global trackFrame;
	global retried_res;
	perframeTimes_s = time.time();
	numberofframes = numberofframes + 1;

	if(ntracker == 0):
		ntracker = local_tracker;
	



	frame_index = frame_index + 1;
	trackFrame = trackFrame + 1;
	#frame = imutils.resize(frame, width=200)



	if (numberofframes == 1):
		#imagename = SCRATCH_DIR + args["directory_output"] + "image_dir/frame_diff_" +str(args["period"]) + str(numberofframes) + "_.bmp"
		#cv2.imwrite(imagename, frame);
		print "Initialize global"
		prevframe_global = copy.deepcopy(frame);
			

	# Start timer
	timer = cv2.getTickCount()
	
	# ok true means frame is differnt diff
	if args["framediffdetector"]:
		ok = framediff(prevframe_global, frame, numberofframes, prevframe_index_global);
	else:
		ok = False;
	
	if ok :
		print "Frame " + str(numberofframes) + " is different from frame " + str(prevframe_index_global);
		prevframe_global = copy.deepcopy(frame);
		prevframe_index_global =  numberofframes



	#ok = False
	bbox = (0,0,0,0)

	# Update tracker
	bbox_arr = []
	ok_arr = []
	ok_arr2 = []
	bbox_arr2 = []
	confidence_arr = [];
	index_arr = [];
	myProcess = psutil.Process(os.getpid())
	myProcess.cpu_percent();
	
	if (not ok) and ((not args["baseline"]) and not (args["t_periodic"] and trackFrame >= args["period"])):
		independentTrackingTimes = independentTrackingTimes + 1
		for i in range (0, ntracker):
			
			tracktime_s = time.time();
			ok, bbox = object_tracking(frame, tracker[i]);
			tracktime_e = time.time();
			trackingTimes = trackingTimes + (tracktime_e - tracktime_s)
			numberoftracking = numberoftracking + 1;
			print "Total time for tracking one object = " + str(tracktime_e - tracktime_s);

			ok_arr.append(True)
			bbox_arr.append(bbox)
	else:
		ok_arr.append(False);
		ntracker = 1;

	#w = args["width"];


	#print "old ntracker " + str(ntracker)
	for i in range(0, ntracker):
		if not ok_arr[i] or args["baseline"] or (args["t_periodic"] and trackFrame >= args["period"]):
			
			client_server_s = time.time();
			frame_to_send = imutils.resize(frame, width=args["intermediate_width"]);
			frame_to_send = cv2.resize(frame_to_send, (args["width"], args["height"]))
			bbox_label = object_detection(s, frame_to_send, numberofframes-1, bbox_data);
			
			#imagename = SCRATCH_DIR + args["directory_output"] + "image_dir/frame_diff_" +str(args["period"]) + str(numberofframes) + "_.bmp"
			#cv2.imwrite(imagename, frame);
	
			if bbox_label.shape == 0:
				return frame;			

		
			s_retry = False;

			for i in range(0, bbox_label.shape[0]):
				if bbox_label[i].size == 1:
					s_retry = True;
					break;

				else:				
					if bbox_label[i][5] <= 0.8 and bbox_label[i][5] >= 0.3:
						s_retry = True;
						break;

			if s_retry and args["resolutionretried"]:
				retried_res = retried_res  + 1
				frame_to_send = imutils.resize(frame, width=400);
				frame_to_send = cv2.resize( frame_to_send, (300, 300) )

				bbox_label = object_detection(s, frame_to_send, numberofframes-1, bbox_data);
					

			client_server_e = time.time();

			for i in range(0, bbox_label.shape[0]):
				print "bbox label--------------------"
				print frame.shape
				print bbox_label[i]
				print bbox_label[i].size
				print type(bbox_label[i])
				if bbox_label[i].size == 1:
					bbox = (0,0,0,0)
					confidence_arr.append(0.3);
					index_arr.append(0)

				else:
					print "bbox label--------------------"

					if (bbox_label[i][1] < 1):
						bbox_label[i][1] =1;
					if (bbox_label[i][0] < 1):
						bbox_label[i][0] =1;

					if (bbox_label[i][1] +bbox_label[i][3] >= frame.shape[0]):
						bbox_label[i][3] = frame.shape[0] - bbox_label[i][1] - 2
					if (bbox_label[i][0] +bbox_label[i][2] >= frame.shape[1]):
						bbox_label[i][2] = frame.shape[1] - bbox_label[i][0] - 2
					

					print "bbo x label--------------------new "
					print frame.shape
					print bbox_label[i]
					print "bbox label-------------------- new"
	 				
					bbox = (bbox_label[i][0], bbox_label[i][1], bbox_label[i][2], bbox_label[i][3])
					confidence_arr.append(bbox_label[i][5]);
					index_arr.append(bbox_label[i][4])

				bbox_arr2.append(bbox);
				ok_arr2.append(True);
				
 				tracker[i].init(frame, bbox)
 				ok, bbox_tmp = object_tracking(frame, tracker[i]);

			#client_server_e = time.time();
			objdetectionTimes = objdetectionTimes + (client_server_e - client_server_s);
			numberofobjdetection = numberofobjdetection + 1;
			
			print "Total time for client server = " + str(client_server_e - client_server_s)

			ntracker = bbox_label.shape[0]
			#print "new ntracker " + str(ntracker)
			
			if trackFrame >= args["period"]:
				#print time.time()
				trackFrame = 0;

			ok_arr = ok_arr2;
			bbox_arr = bbox_arr2;

			break;


	# Calculate Frames per second (FPS)

	fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
	# Draw bounding box

	#trackTimer = trackTimer + 1/fps
	
	if(len(confidence_arr) != 0):
		confidence_arr_global = confidence_arr;
		index_arr_global = index_arr;


	print "current ntracker" + str(ntracker)
	for i in range(0, ntracker):
		if ok_arr[i]:
			# Tracking success
			p1 = (int(bbox_arr[i][0]), int(bbox_arr[i][1]))
			p2 = (int(bbox_arr[i][0] + bbox_arr[i][2]), int(bbox_arr[i][1] + bbox_arr[i][3]))
			cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

			# calculate area 
			bbox_area = (bbox_arr[i][2]) * (bbox_arr[i][3]);

			outputfile.write(str(i) + "," + str(bbox_arr[i][0]) + "," + str(bbox_arr[i][1]) + "," + str(bbox_arr[i][0] + bbox_arr[i][2]) + "," + str(bbox_arr[i][1] + bbox_arr[i][3]));

			# Display class label 
			if len(confidence_arr_global) != 0: 
				startX = int ( bbox_arr[i][0] );
				y = int (bbox_arr[i][1] - 15) if bbox_arr[i][1] - 15 > 15 else int(bbox_arr[i][1] + 15)
				#label = str(bbox_area);
				label = str(CLASSES[int(index_arr_global[i])]) + " - " +str(confidence_arr_global[i]) + "%";
				cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[0], 2)

				outputfile.write("," + str(index_arr_global[i]) + "," + str(confidence_arr_global[i]))

			if i == ntracker - 1:
				continue;
			else:
				outputfile.write("|")

		else :
			# Tracking failure
			cv2.putText(frame, "Failure in object detection and tracking", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

	cpupercent = myProcess.cpu_percent()
	print cpupercent;
	cpupercent_total = cpupercent_total + cpupercent;
	memorypercent_total = memorypercent_total + myProcess.memory_percent()

	# Display FPS on frame
	#cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
	outputfile.write("\n");	
	
	# Display result
	perframeTimes_e = time.time();
	perframeTimes = perframeTimes + perframeTimes_e - perframeTimes_s;

	imagename = SCRATCH_DIR + args["directory_output"] + "image_dir/" +str(args["period"]) + str(numberofframes) + "_.bmp"
	print imagename
	cv2.imwrite(imagename, frame);

	#cv2.imshow('image',frame)
	#k = cv2.waitKey(0)


	return frame;
####################################################################################################
def object_detection(s, frame, iframe, bbox_data):
	if args["readfrommemory"]:
		bbox_list = [];
		bbox_array = np.array(bbox_data[iframe].split("|"));
		for ii in range(0, bbox_array.shape[0]):
			bbox_array_element = bbox_array[ii].split(",");
			bbox_list.append([float(bbox_array_element[1]),float(bbox_array_element[2]),float(bbox_array_element[3]) - float(bbox_array_element[1]),float(bbox_array_element[4])-float(bbox_array_element[2]), float(bbox_array_element[5]), float(bbox_array_element[6])])
		
		bbox_list = np.array(bbox_list)
	
		return bbox_list;
	else:

		#for i in range (0, bbox_list.shape[0]):
		#	p1 = (int(bbox_list[i][0]), int(bbox_list[i][1]))
		#	p2 = (int(bbox_list[i][0] + bbox_list[i][2]), int(bbox_list[i][1] + bbox_list[i][3]))
		#	cv2.rectangle(frame, p1, p2, (0,0,255), 2, 1)

		if args["local"]:
			return l_object_detection(frame);
		else:
			return s_object_detection(s, frame);


#####################################################################################################
# construct the argument parse and parse the arguments
if __name__ == "__main__":

	# If want to read from file
	file_baseline  = open(args["basefile"], "r");
	bbox_data = np.array(file_baseline.readlines());

	myStartTime = time.time();

	white_output = SCRATCH_DIR + args["directory_output"] + args["output_video"]
	clip1 = VideoFileClip(SCRATCH_DIR + args["input_video"]);

	frame = clip1.get_frame(0);
	#frame = imutils.resize(frame, width=200);

	#prevframe = frame;
	#bbox = cv2.selectROI(frame, False)
	

	initTime_s = time.time();
	ctx = zmq.Context.instance();
	s = ctx.socket(zmq.REQ)
	addr = 'tcp://' +  args["server"]
	s.connect(addr)
	
	frame_to_send = imutils.resize(frame, width=args["intermediate_width"]);
	frame_to_send = cv2.resize(frame_to_send, (args["width"], args["height"]))
	bbox_label = object_detection(s, frame_to_send, 0, bbox_data);

	w = args["width"];
	
	for i in range(0, bbox_label.shape[0]):
		bbox = (bbox_label[i][0], bbox_label[i][1], bbox_label[i][2], bbox_label[i][3])
		tracker[i].init(frame, bbox)
	initTime_e = time.time();	

	outputfilename = os.path.join(SCRATCH_DIR + args["directory_output"], args["file_output"]);
	outputfile = open(outputfilename, "w");

	local_tracker = bbox_label.shape[0]
	label = ""
	white_clip = clip1.fl_image(lambda x: frameprocessing(x, label, local_tracker, s, outputfile, bbox_data))
	white_clip.write_videofile(white_output, audio=False)
	myEndTime = time.time();

	outputfile.flush();
	outputfile.close();

	s.disconnect(addr)
	s.close();
	



	sizedata = frame.shape[0]*frame.shape[1]*frame.shape[2]*4
	
	if totalTime != 0:
		print "Bandwidth : " + str( float(totalSendRecv * sizedata) / (totalTime * 1024*1024)) + " MByte/s"
	print "Cpu percent : " + str(cpupercent_total/numberofframes);
	print "Memory percent : " + str(memorypercent_total/numberofframes);
	print "Time : " + str(myEndTime - myStartTime);

	print "Server Client data : " + str(serverclienttotalBytes);
	print "Client Server data : " + str(clienserverttotalBytes);

	print "Total Time for object detection : " + str(objdetectionTimes);
	if(numberofobjdetection != 0):
		print "Average time for object_detection : " + str(objdetectionTimes/numberofobjdetection)
	
	print "Total number object_detection : " + str(numberofobjdetection)

	print "Total Time for tracking : " + str(trackingTimes);
	if (numberoftracking !=0):
		print "Average time for tracking : " + str(trackingTimes/numberoftracking);
	print "Total number tracking " + str(numberoftracking)
	print "Total number of doing tracking (independent of number of objects) " + str(independentTrackingTimes)
	print "Number of retried_res : " + str(retried_res)
	print "Number of frames : " + str(numberofframes)
	print "Total Perframe times : " + str(perframeTimes);
	if(numberofframes !=0):
		print "Averafe perframe times : " + str(perframeTimes/numberofframes)
	
	if(perframeTimes != 0):
		print "FrameRate : " + str(numberofframes/perframeTimes)
	
	print "Resolution %d x %d x 3" % (args["width"], args["height"]);
	print "Total data sent : " + str(args["width"] * args["height"] * 3 * numberofframes + 300 * 300 * 3 * retried_res)
	
	print "Total data sent intermediate : " + str(args["intermediate_width"] * int(360 / (640.0 / args["intermediate_width"])) * 3 * numberofframes + 400 * 225 * 3 * retried_res)
	#if not args["baseline"]:
	print "Accuracy : " + str(check_IOU.calculate_IOU(args["basefile"], outputfilename, args["directory_output"]));
	print "Initialize time : " + str(initTime_e - initTime_s);

	sys.exit();