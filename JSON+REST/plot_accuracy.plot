set term postscript eps color 25 font ",28"
set xlabel '# of frames sent' 
set ylabel 'Percentage (%)'
set xrange [0:192]
set yrange [0.6:1]



set ytics ("1"  1, ".9" 0.9, ".8" 0.8,  ".7" 0.7, ".6" 0.6)
set key bot right
set size 1,1
#set output 'eps/chaplin-number-frames-accuracy-period.eps'

#plot \
#'scratch_data_chaplin/chaplin_periodicity_accuracy.dat'     u ($1):2 t 'F1' w l lw 8 lc  rgb "green", \
#'scratch_data_chaplin/chaplin_periodicity_accuracy.dat'  u ($1):3 t 'Precision' w l lw 4 lc  rgb "red" , \
#'scratch_data_chaplin/chaplin_periodicity_accuracy.dat'   u ($1):4 t 'Recall' w l  lw 2 lc  rgb "blue" 

#set output 'eps/chaplin-number-frames-accuracy-framediff.eps'

#plot \
#'scratch_data_chaplin/chaplin_difference_detector_2_accuracy.dat'     u ($1):2 t 'F1' w l lw 3 lc  rgb "green" , \
#'scratch_data_chaplin/chaplin_difference_detector_2_accuracy.dat'  u ($1):3 t 'Precision' w l lw 3 lc  rgb "red" , \
#'scratch_data_chaplin/chaplin_difference_detector_2_accuracy.dat'   u ($1):4 t 'Recall' w l  lw 3 lc  rgb "blue" 

set output 'eps/chaplin_period_vs_diffdetect_recall.pdf'

plot \
'scratch_data_chaplin/chaplin_periodicity_accuracy.dat'  u ($1):4 t 'Recall-Period' w l lw 4 lc  rgb "red" , \
'scratch_data_chaplin/chaplin_difference_detector_2_accuracy.dat'   u ($1):4 t 'Recall-Diffdetector' w l  lw 2 lc  rgb "blue"

set output 'eps/chaplin_period_vs_diffdetect_F1.eps'

plot \
'scratch_data_chaplin/chaplin_periodicity_accuracy.dat'  u ($1):2 t 'F1-Period' w l lw 4 lc  rgb "red" , \
'scratch_data_chaplin/chaplin_difference_detector_2_accuracy.dat'   u ($1):2 t 'F1-Diffdetector' w l  lw 2 lc  rgb "blue"

set output 'eps/chaplin_period_vs_diffdetect_precision.pdf'

plot \
'scratch_data_chaplin/chaplin_periodicity_accuracy.dat'  u ($1):3 t 'Precision-Period' w l lw 4 lc  rgb "red" , \
'scratch_data_chaplin/chaplin_difference_detector_2_accuracy.dat'   u ($1):3 t 'Precision-Diffdetector' w l  lw 2 lc  rgb "blue"