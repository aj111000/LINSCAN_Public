The cross-validation experiment to compare LINSCAN to OPTICS is in "linscan_optics_comparison.py". To run LINSCAN, linscan_c.c has to be compiled to a format which can be used. To do this, run the following line using the GNU compiler collection:

gcc -fPIC -shared -o [LINSCAN_Public Path]\linscan_c.so [LINSCAN_Public Path]\linscan_c.c