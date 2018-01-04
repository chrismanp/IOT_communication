set term postscript eps color 25 font ",24"
set xlabel 'Data sent (Kbyte)' 
set ylabel 'Percentage (0-1)'
set xrange [0:250000]
set yrange [0.6:1.1]



set ytics ("1"  1, ".9" 0.9, ".8" 0.8,  ".7" 0.7, ".6" 0.6)
set xtics 50000
set key top right
set size 1,1

set output 'eps/chaplin_resolution_intermediatestep_F1.eps'

plot \
'scratch_data_imageresolution_2/resolution_accuracy_retry.dat'  u ($1/1024):2 t 'F1-WithRetry' w lp lw 4 lc  rgb "red" , \
'scratch_data_imageresolution_2/resolution_accuracy_noretry.dat'   u ($1/1024):2 t 'F1-NoRetry' w lp  lw 2 lc  rgb "blue", \
'scratch_data_imageresolution_2/guideline.dat'   u ($1/1024):2 t 'Baseline' w l lw 2 lc  rgb "green"

set output 'eps/chaplin_resolution_intermediatestep_precision.pdf'

plot \
'scratch_data_imageresolution_2/resolution_accuracy_retry.dat'  u ($1/1024):3 t 'Precision-WithRetry' w p lw 4 lc  rgb "red" , \
'scratch_data_imageresolution_2/resolution_accuracy_noretry.dat'   u ($1/1024):3 t 'Precision-NoRetry' w p  lw 2 lc  rgb "blue", \
'scratch_data_imageresolution_2/guideline.dat'   u ($1/1024):2 t 'Baseline' w l lw 2 lc  rgb "green"


set output 'eps/chaplin_resolution_intermediatestep_recall.pdf'

plot \
'scratch_data_imageresolution_2/resolution_accuracy_retry.dat'  u ($1/1024):4 t 'Recall-WithRetry' w p lw 4 lc  rgb "red" , \
'scratch_data_imageresolution_2/resolution_accuracy_noretry.dat'   u ($1/1024):4 t 'Recall-NoRetry' w p  lw 2 lc  rgb "blue", \
'scratch_data_imageresolution_2/guideline.dat'   u ($1/1024):2 t 'Baseline' w l lw 2 lc  rgb "green"


##############################################################################################################

set output 'eps/chaplin_resolution_direct_F1.eps'

plot \
'scratch_data_imageresolution_no_300_300_2/resolution_accuracy_retry.dat'  u ($1/1024):2 t 'F1-WithRetry' w lp lw 4 lc  rgb "red" , \
'scratch_data_imageresolution_no_300_300_2/resolution_accuracy_noretry.dat'   u ($1/1024):2 t 'F1-NoRetry' w lp  lw 2 lc  rgb "blue", \
'scratch_data_imageresolution_no_300_300_2/guideline.dat'   u ($1/1024):2 t 'Baseline' w l lw 2 lc  rgb "green"

set output 'eps/chaplin_resolution_direct_precision.pdf'

plot \
'scratch_data_imageresolution_no_300_300_2/resolution_accuracy_retry.dat'  u ($1/1024):3 t 'Precision-WithRetry' w p lw 4 lc  rgb "red" , \
'scratch_data_imageresolution_no_300_300_2/resolution_accuracy_noretry.dat'   u ($1/1024):3 t 'Precision-NoRetry' w p  lw 2 lc  rgb "blue", \
'scratch_data_imageresolution_no_300_300_2/guideline.dat'   u ($1/1024):2 t 'Baseline' w l lw 2 lc  rgb "green"

set output 'eps/chaplin_resolution_direct_recall.pdf'

plot \
'scratch_data_imageresolution_no_300_300_2/resolution_accuracy_retry.dat'  u ($1/1024):4 t 'Recall-WithRetry' w p lw 4 lc  rgb "red" , \
'scratch_data_imageresolution_no_300_300_2/resolution_accuracy_noretry.dat'   u ($1/1024):4 t 'Recall-NoRetry' w p  lw 2 lc  rgb "blue", \
'scratch_data_imageresolution_no_300_300_2/guideline.dat'   u ($1/1024):2 t 'Baseline' w l lw 2 lc  rgb "green"

###############################################################################################################
#scratch_data_image_resolution_on_objdetection

set output 'eps/chaplin_resolution_400direct_F1.eps'

plot \
'scratch_data_image_resolution_on_objdetection/resolution_accuracy_retry.dat'  u ($1/1024):2 t 'F1-WithRetry' w lp lw 4 lc  rgb "red" , \
'scratch_data_image_resolution_on_objdetection/resolution_accuracy_noretry.dat'   u ($1/1024):2 t 'F1-NoRetry' w lp  lw 2 lc  rgb "blue", \
'scratch_data_image_resolution_on_objdetection/guideline.dat'   u ($1/1024):2 t 'Baseline' w l lw 2 lc  rgb "green"

set output 'eps/chaplin_resolution_400direct_precision.pdf'

plot \
'scratch_data_image_resolution_on_objdetection/resolution_accuracy_retry.dat'  u ($1/1024):3 t 'Precision-WithRetry' w p lw 4 lc  rgb "red" , \
'scratch_data_image_resolution_on_objdetection/resolution_accuracy_noretry.dat'   u ($1/1024):3 t 'Precision-NoRetry' w p  lw 2 lc  rgb "blue", \
'scratch_data_image_resolution_on_objdetection/guideline.dat'   u ($1/1024):2 t 'Baseline' w l lw 2 lc  rgb "green"

set output 'eps/chaplin_resolution_400direct_recall.pdf'

plot \
'scratch_data_image_resolution_on_objdetection/resolution_accuracy_retry.dat'  u ($1/1024):4 t 'Recall-WithRetry' w p lw 4 lc  rgb "red" , \
'scratch_data_image_resolution_on_objdetection/resolution_accuracy_noretry.dat'   u ($1/1024):4 t 'Recall-NoRetry' w p  lw 2 lc  rgb "blue", \
'scratch_data_image_resolution_on_objdetection/guideline.dat'   u ($1/1024):2 t 'Baseline' w l lw 2 lc  rgb "green"
