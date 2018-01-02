import os
import sys



if __name__ == "__main__":
	for i in [1, 2, 3, 4, 5, 8, 10, 20, 60, 100, 192]:	

		os.system("mkdir scratch_data/output_dir_chaplin_%d/image_dir -p" %(i, )) 

		os.system('python client_detection_and_tracking.py -p MobileNetSSD_deploy.prototxt.txt -m MobileNetSSD_deploy.caffemodel -i chaplin.mp4 -o chaplin_out.mp4 -d output_dir_chaplin_%d/ -b 0 -f chaplin_o_%d.log -t 1 -u %d -r 0 -z scratch_data/output_dir1/chaplin_1.log -y 0 -x 1 -k 35000 > scratch_data/output_dir_chaplin_%d/result.log' %(i, i, i, i))

