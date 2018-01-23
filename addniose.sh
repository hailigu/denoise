#!/bin/bash 
#Noise_Addition
#This script use sox to add noise to the speech file
#Tue JAN 23 2018 Xugaopeng
for x in ./*.wav;do
    b=${x##*/}  
    sox -m noise.wav $b tmp-$b  
    rm -rf $b  
    mv tmp-$b $b  
done  
