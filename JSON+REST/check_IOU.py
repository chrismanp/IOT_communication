
import os;
import argparse;
import numpy as np;
import matplotlib.pyplot as plt

SCRATCH_DIR = "/home/mnt_sdc/scratch_data/"

def calculate_IOU(file1_txt, file2_txt, outputdir):
	#print "file1_txt : " + file1_txt
	#print "file2_txt : " + file2_txt 
	cntfilename = file2_txt.split(".")[0] + "_positive_negative_cnt.log"


	file1 = open(file1_txt, "r");
	file2 = open(file2_txt, "r");
	cntfile = open(cntfilename, "w");

	
	cnt = 0;
	idx = 0;
	precision_total = 0;
	precision_sum = [];
	recall_sum = [];
	recall_total = 0;

	precision_overtime = 1;
	recall_overtime = 1;

	tp = [];
	fp = [];
	#tn = [];
	fn = [];
	truthvalue = [];

	cntfile.write("Frame" + "," + "TruePositive" +"," + "FalsePositive" + "," + "FalseNegative"+ ","+ "#ProposedRegion" + "," + "#ActualRegion" + "\n");

	line2_list = file2.readlines()
	for line1 in iter(file1):

		#if (idx > 100):
		#	break;


		cnttrue = 0;

		line2 = line2_list[idx]
			

		idx = idx + 1;	


		line1_bigarr = line1.split("|")
		line2_bigarr = line2.split("|")

		allresult =  len(line2_bigarr)
		trueresult = len(line1_bigarr)

		truthvalue.append(trueresult);
		ismatched1 = np.zeros(len(line1_bigarr))
		ismatched2 = np.zeros(len(line2_bigarr))

		for ii in range(0, len(line1_bigarr)):
			for jj in range(0, len(line2_bigarr)):
				
				if(ismatched1[ii] == 1 or ismatched2[jj] == 1):
					continue;

				line1_bigarr_element = line1_bigarr[ii]
				line2_bigarr_element = line2_bigarr[jj]

				line1_arr2 = line1_bigarr_element.split(",")
				line2_arr2 = line2_bigarr_element.split(",")

				#if(line1_arr[4] != line2_arr[4]):
				#	continue;


				

				line1_arr = [float(x) for x in line1_arr2[1:5]]
				line2_arr = [float(x) for x in line2_arr2[1:5]]


				x_1 = max(line1_arr[0], line2_arr[0]);
				x_2 = min(line1_arr[2], line2_arr[2]);
				y_1 = max(line1_arr[1], line2_arr[1]);
				y_2 = min(line1_arr[3], line2_arr[3]);


				if(y_2 - y_1 <= 0):
					continue;
				if(x_2 - x_1 <= 0):
					continue;

				area_intersection = (y_2 - y_1) * (x_2 - x_1)
				print "line1_arr"
				print line1_arr
				print line2_arr
				print "area : "
				print area_intersection
				#print "-----------------------"
				#print (area_intersection)

				area_union = (line1_arr[3] - line1_arr[1]) * (line1_arr[2] - line1_arr[0]) 
				+ (line2_arr[3] - line2_arr[1]) * (line2_arr[2] - line2_arr[0])
				- area_intersection;

				#print (area_union)
				#print "-----------------------"

				IOU = area_intersection/area_union;
				if IOU >= 0.3:
					cnttrue = cnttrue + 1
					ismatched1[ii] = 1;
					ismatched2[jj] = 1;
		
		tp.append(cnttrue);
		fp.append(allresult-cnttrue);
		fn.append(trueresult-cnttrue);


		precision = float(cnttrue)/allresult
		print str(idx) + " Precision  : " + str(precision)
		recall    = float(cnttrue)/trueresult
		print str(idx) + " Recall : " + str(recall);

		#precision_overtime = 0.5 * precision_overtime + 0.5 * precision
		#recall_overtime = 0.5 * recall_overtime + 0.5 * recall

		precision_total = precision_total + precision;
		precision_sum.append(precision_total);
		recall_total = recall_total + recall;
		recall_sum.append(recall_total);
		cntfile.write(str(idx) + "," + str(cnttrue) +"," + str(allresult-cnttrue) + "," + str(trueresult-cnttrue)+ ","+str(allresult) + "," +str(trueresult) + "\n")
		print str(idx) + " Precision over time : " + str(precision_total)
		#recall    = float(cnttrue)/trueresult
		print str(idx) + " Recall over time : " + str(recall_total); 

		
			

	precision_avg = precision_total/idx;
	print "Precision avg = " + str(precision_avg)
	recall_avg = recall_total/idx;	
	print "Recall avg = " + str(recall_avg);

	F1_score = 2.0 * precision_avg * recall_avg /(precision_avg + recall_avg);

	# Plot the image
	
	plt.title("Frame vs Metric")
	plt.xlabel("Frame")
	plt.ylabel("Count")

	#plt.plot(range(1, idx+1), tp, 'bo', label="True Positive")
	plt.plot(range(1, idx+1), fn, 'go', label="False Negative");
	plt.plot(range(1, idx+1), fp, 'ro', label="False Positive")

	plt.legend(loc="best")
	pdffilename = file2_txt.split(".")[0] + ".pdf"
	print pdffilename;
	plt.savefig(pdffilename)
	#plt.show();

	plt.figure()
	plt.axis()
	plt.title("Frame vs Sum of precison/recall")
	plt.xlabel("Frame")
	plt.ylabel("Count")

	#plt.plot(range(1, idx+1), tp, 'bo', label="True Positive")
	plt.plot(range(1, idx+1), precision_sum, 'g', label="Precision");
	plt.plot(range(1, idx+1), recall_sum, 'r', label="Recall")

	plt.legend(loc="best")
	pdffilename = file2_txt.split(".")[0] +"precision_recall" + ".pdf"
	print pdffilename;
	plt.savefig(pdffilename)
	#plt.show();
	

	file1.close();
	file2.close();
	cntfile.close();

	#print "cnt : " + str(cnt );
	#return float(cnttrue)/cnt	
	return F1_score

if __name__ == "__main__":

	ap = argparse.ArgumentParser()
	ap.add_argument("-b", "--baselinefile", required=True,
	help="path to baseline file")
	ap.add_argument("-c", "--comparefile", required=True,
	help="path to file to compare")
	args = vars(ap.parse_args())

	#calculate_IOU(args["baselinefile", args["comparefile"]);

