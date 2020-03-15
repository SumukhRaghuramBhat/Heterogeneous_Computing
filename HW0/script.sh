#! /bin/bash
#PBS -q mamba
#PBS -l nodes=1:ppn=1:gpus=1
./GPU.out
