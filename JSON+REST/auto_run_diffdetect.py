import os
import sys



if __name__ == "__main__":
	for i in [500, 1000, 2000, 3000, 4000, 5000, 8000, 12000, 15000 ]:	

		os.system("mkdir scratch_data/output_dir_%d_sumthresh2/image_dir -p" %(i, )) 

		os.system('python client_detection_and_tracking.py -p MobileNetSSD_deploy.prototxt.txt -m MobileNetSSD_deploy.caffemodel -i chaplin.mp4 -o chaplin_o.mp4 -d output_dir_%d_sumthresh2/ -b 0 -f chaplin_%d_sumthresh.log -t 0 -u 1 -r 0 -z scratch_data/output_dir1/chaplin_1.log -x 100 -k %d -y 1 > scratch_data/output_dir_%d_sumthresh2/result.log' %(i, i, i, i))

