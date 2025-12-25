#!/bin/sh
# This script will run the CNN models application and generate a memory trace log file

models_directory="./models"
input_data_path="./data/"
labels_path="./models/imagenet_classes.txt"
shared_library_directory="../../tools/mem_trace"
#log_folder="./log/Raw"
log_folder="/peta_mda1/sg05060/MPC/0_dataset/log2/Raw"
modifiedlog_folder="/peta_mda1/sg05060/MPC/0_dataset/log2/OptimizedCacheLayout"
Loader_folder="../../Loader/for_mem_trace"
Loader="../../Loader/for_mem_trace/main.out"
numpy_Loader="../../Loader/for_mem_trace/Loader.py"
flag=0
echo "script Start"

if [ ! -d "$log_folder" ]; then
  mkdir -p "$log_folder"
  echo "폴더가 생성되었습니다: $log_folder"
else
  echo "폴더가 이미 존재합니다: $log_folder"
fi

if [ ! -d "$modifiedlog_folder" ]; then
  mkdir -p "$modifiedlog_folder"
  echo "폴더가 생성되었습니다: $modifiedlog_folder"
else
  echo "폴더가 이미 존재합니다: $modifiedlog_folder"
fi

# Makes shared Library
cd $shared_library_directory
make
cd -

# Makes Loader
cd $Loader_folder
g++ -std=c++11 -o main.out for_mem_trace.cpp 
cd -

## Run models and extract log file for mem_trace
#for file in "$models_directory"/*.py; do
#    if [ -f "$file" ]; then
#        log_file="$log_folder/$(basename "$file" .py)_log.txt"
#
#        if [ -f "$log_file" ]; then
#            echo "$log_file already exists. Skipping $file..."
#        else
#            echo "Executing $file..."
#            LD_PRELOAD="$shared_library_directory/mem_trace.so" python "$file" $input_data_path --labels $labels_path >> "$log_file"
#
#            chmod 777 "$log_file"
#        fi
#    fi  
#done
#
#
## Aggregate log file to optimize Cache line size
#for file in "$log_folder"/*.txt; do
#    if [ -f "$file" ]; then
#        modifedlog_file="$modifiedlog_folder/$(basename "$file" .txt)_modified.txt"
#
#        if [ -f "$modifedlog_file" ]; then
#            echo "$modifedlog_file already exists. Skipping $file..."
#        else
#            echo "Aggregating $file..."
#            $Loader $file $modifedlog_file
#
#        fi
#    fi  
#done
#
#
## modify text file to npy file
#for file in "$modifiedlog_folder"/*.txt; do
#    if [ -f "$file" ]; then
#        npy_file="$modifiedlog_folder/$(basename "$file" .txt).npy"
#
#        if [ -f "$npy_file" ]; then
#            echo "$npy_file already exists. Skipping $file..."
#        else
#            echo "Modify text to numpy $file..."
#            python $numpy_Loader $file $npy_file
#            echo "Modify text to numpy $file. Exit code: $?" >> "$npy_file"
#            echo "" >> "$npy_file"
#        fi
#    fi  
#done



#############Test############
for file in "$models_directory"/*.py; do
    if [ -f "$file" ]; then
        # Make log file
        log_file="$log_folder/$(basename "$file" .py)_log.txt"
        if [ -f "$log_file" ]; then
            echo "$log_file already exists. Skipping $file..."
        else
            flag=0
            for files in $modifiedlog_folder; do
                if [[ "$files" == *"$file"* ]]; then
                    echo "파일 이름에 '$file_pattern' 패턴이 포함된 파일이 $folder_path 폴더에 존재합니다."
                    flag=1
                    break
                fi
            done
            if [[ "$flag" -eq 0 ]]; then
             echo "Executing $file..."
             LD_PRELOAD="$shared_library_directory/mem_trace.so" python "$file" $input_data_path --labels $labels_path >> "$log_file"
            fi
        fi
        chmod 777 "$log_file"
        
        # Aggregate log file to optimize Cache line size & Delete origin log file
        modifedlog_file="$modifiedlog_folder/$(basename "$log_file" .txt)_modified.txt"
        if [ -f "$modifedlog_file" ]; then
            echo "$modifedlog_file already exists. Skipping $log_file..."
        else
            flag=0
            for files in $modifiedlog_folder; do
                if [[ "$files" == *"$file"* ]]; then
                    echo "파일 이름에 '$file_pattern' 패턴이 포함된 파일이 $folder_path 폴더에 존재합니다."
                    flag=1
                    break
                fi
            done
            if [[ "$flag" -eq 0 ]]; then
             echo "Aggregating $log_file..."
             $Loader $log_file $modifedlog_file
            fi
        fi

        if [ -f "$log_file" ]; then
          rm $log_file
        fi

        # Modify log text file to npy file
        npy_file="$modifiedlog_folder/$(basename "$modifedlog_file" .txt).npy"

        if [ -f "$npy_file" ]; then
            echo "$npy_file already exists. Skipping $modifedlog_file..."
        else
            echo "Modify text to numpy $modifedlog_file..."
            python $numpy_Loader $modifedlog_file $npy_file
        fi

        if [ -f "$modifedlog_file" ]; then
          rm $modifedlog_file
        fi

    fi  
done
#############################