import numpy as np

# 저장된 배열 파일 읽기
loaded_array = np.load("/home/sg05060/CUDA/NVBit_Profile_Data/test-apps/CNN_model/log/OptimizedCacheLayout/resnet18_log_modified.npy")

# 배열 크기 출력
print(f"Shape: {loaded_array.shape}")

# 배열 내용 출력
for row in loaded_array:
    print(" ".join(map(str, row)))
