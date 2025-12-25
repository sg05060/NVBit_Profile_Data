#!/bin/sh
# This script will run the vectoradd application and generate a memory trace log file
echo "script Start"
echo "vetoradd make"
make
cd ../../tools/mem_trace/
make
cd ../../test-apps/vectoradd/
./vectoradd
echo "mem_trace_log"
LD_PRELOAD=../../tools/mem_trace/mem_trace.so ./vectoradd > mem_trace_log.txt &
echo "record_reg_val"
LD_PRELOAD=../../tools/record_reg_vals/record_reg_vals.so ./vectoradd > reg_val_log.txt &
cd /home/sg05060/CUDA/nvbit_release/Loader/for_mem_trace
./for_mem_trace.exe
echo "script done"