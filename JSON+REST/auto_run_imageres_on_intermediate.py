import os
import sys



if __name__ == "__main__":
	for i in [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600 ,640, 800, 1280]:	

		print ("%d no retry procesing" %(i, ))

		os.system("mkdir scratch_data/output_dir_chaplin_intermediateres_%d/image_dir -p" %(i, )) 

		os.system('python client_image_resolution.py -p MobileNetSSD_deploy.prototxt.txt -m MobileNetSSD_deploy.caffemodel -i chaplin.mp4 -o chaplin_o.mp4 -d output_dir_chaplin_intermediateres_%d/ -b 0 -f chaplin_w%d.log -l 0 -t 1 -z scratch_data/output_dir_chaplin_highres_400_2/chaplin_w400.log -u 1 -r 0 -x 0 -k 0 -y 0 -w 300 -n 300 -e %d -v 0 > scratch_data/output_dir_chaplin_intermediateres_%d/result.log' %(i, i, i, i))

	for i in [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600 ,640, 800, 1280]:	

		print ("%d retry procesing" %(i, ))

		os.system("mkdir scratch_data/output_dir_chaplin_intermediateretryres_%d/image_dir -p" %(i, )) 

		os.system('python client_image_resolution.py -p MobileNetSSD_deploy.prototxt.txt -m MobileNetSSD_deploy.caffemodel -i chaplin.mp4 -o chaplin_o.mp4 -d output_dir_chaplin_intermediateretryres_%d/ -b 0 -f chaplin_w%d.log -l 0 -t 1 -z scratch_data/output_dir_chaplin_highres_400_2/chaplin_w400.log -u 1 -r 0 -x 0 -k 0 -y 0 -w 300 -n 300 -e %d -v 1 > scratch_data/output_dir_chaplin_intermediateretryres_%d/result.log' %(i, i, i, i))

