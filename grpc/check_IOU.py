
import os;
import argparse;

def calculate_IOU(file1_txt, file2_txt):
	#print "file1_txt : " + file1_txt
	#print "file2_txt : " + file2_txt 


	file1 = open(file1_txt, "r");
	file2 = open(file2_txt, "r");


	cnttrue = 0;
	cnt = 0;
	line2_list = file2.readlines()
	for line1 in iter(file1):

		line2 = line2_list[cnt]
		cnt = cnt + 1;
			

		line1_arr = line1.split(",")
		line2_arr = line2.split(",")

		if(line1_arr[4] != line2_arr[4]):
			continue;

		line1_arr = [float(x) for x in line1_arr[0:4]]
		line2_arr = [float(x) for x in line2_arr[0:4]]

		x_1 = max(line1_arr[0], line2_arr[0]);
		x_2 = min(line1_arr[2], line2_arr[2]);
		y_1 = max(line1_arr[1], line2_arr[1]);
		y_2 = min(line1_arr[3], line2_arr[3]);

		area_intersection = (y_2 - y_1) * (x_2 - x_1)

		#print "-----------------------"
		#print (area_intersection)

		area_union = (line1_arr[3] - line1_arr[1]) * (line1_arr[2] - line1_arr[0]) 
		+ (line2_arr[3] - line2_arr[1]) * (line2_arr[2] - line2_arr[0])
		- area_intersection;

		#print (area_union)
		#print "-----------------------"

		IOU = area_intersection/area_union;
		if IOU > 0.5:
			cnttrue = cnttrue + 1


	file1.close();
	file2.close();

	if cnt == 0:
		return -1;

	#print "cnt : " + str(cnt );
	return float(cnttrue)/cnt	

if __name__ == "__main__":

	ap = argparse.ArgumentParser()
	ap.add_argument("-b", "--baselinefile", required=True,
	help="path to baseline file")
	ap.add_argument("-c", "--comparefile", required=True,
	help="path to file to compare")
	args = vars(ap.parse_args())

	calculate_IOU(file1, file2);

