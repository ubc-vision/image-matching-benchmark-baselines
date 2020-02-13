#!/bin/bash

for fdir in ../benchmark-features/* 
  do 
     echo ${fdir}
     fname="$(basename "$fdir")"
     echo $fname 
     for seq in ${fdir}/*  
        do  
           seqname="$(basename "$seq")"; 
           mkdir ../results_raw_upright/$seqname/$fname
           cp $seq/* ../results_raw_upright/$seqname/$fname/
        done
done
