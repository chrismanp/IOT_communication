set term postscript eps color 25 font ",24"
set xlabel 'Frame Sequence' 
set ylabel '# of Object'
set xrange [1:191]
set yrange [0:3]



set xtics 30
set key top left
set size 1,1

set output 'eps/series_info.eps'

plot \
'scratch_data_imageresolution/series_info.dat'  u 1:2 t 'redundant' w p ps 1 pt 4 lc  rgb "red", \
'scratch_data_imageresolution/series_info_essential.dat'  u 1:2 t 'important' w p ps 2 pt 7 lc  rgb "blue"