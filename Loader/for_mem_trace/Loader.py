import numpy as np
import argparse

# 명령줄 인자 파싱을 설정
parser = argparse.ArgumentParser(description='텍스트 파일을 NumPy 형식의 파일로 변환하는 스크립트')
parser.add_argument('input_file', type=str, help='입력 텍스트 파일 경로')
parser.add_argument('output_file', type=str, help='출력 NumPy 파일 경로')
args = parser.parse_args()

# 텍스트 파일에서 라인 수를 세어 행 수로 사용
with open(args.input_file, 'r') as data_file:
    num_rows = sum(1 for line in data_file)

num_columns = 32  # 열 수 (고정값)

# 텍스트 파일에서 데이터를 읽어와 NumPy 배열 생성
data_list = []
with open(args.input_file, 'r') as data_file:
    for line in data_file:
        line = line.strip()  # 라인 앞뒤 공백 및 개행 문자 제거
        # 라인을 2개씩 문자로 나누고 정수로 변환하여 리스트에 추가
        data_list.extend(int(line[i:i+2], 16) for i in range(0, len(line), 2))

# 데이터가 부족할 경우 예외 처리
total_elements = num_rows * num_columns
if len(data_list) < total_elements:
    raise ValueError("텍스트 파일에서 가져온 데이터가 부족합니다.")

# NumPy 배열로 변환 (행과 열에 따라 reshape)
data_array = np.array(data_list[:total_elements], dtype=np.uint8).reshape(num_rows, num_columns)
print(data_array)
# NumPy 배열을 NumPy 형식의 파일로 저장
np.save(args.output_file, data_array)

print(f"NumPy 배열을 {args.output_file}로 저장했습니다.")
