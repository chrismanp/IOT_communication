import os
import sys



if __name__ == "__main__":
	for i in [1, 5, 20, 40, 60, 80, 100, 192]:


		os.system('time python client_detection_and_tracking.py --prototxt ../grpc/MobileNetSSD_deploy.prototxt.txt --model ../grpc/MobileNetSSD_deploy.caffemodel -i chaplin.mp4 -o chaplin_%d.mp4 -d outputdir/ -b 0 -f bbox_normal%d.txt -l 0 -t 1 -z outputdir/bbox_baseline.txt -u %d -s "10.1.1.3:9999" -r 0 > %d_information_boost.log4' %(i, i, i, i))

