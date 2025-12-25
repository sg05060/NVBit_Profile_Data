#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <bitset>

//#define DATA_STRING_SIZE 16 // 16 Hex(64bit data) 
#define ADDRESS_STRING_SIZE 16 // 16 Hex(64bit data)
using namespace std;

void HexString_to_BinaryString(const std::string& HexString, std::string& BinaryString);
void BinarySting_to_HexString(const std::string& binaryString, std::string& HexString);

std::string Kernel_start_message = "LAUNCH - Kernel pc";

typedef  union {
    char g_one_bytes[32];
    int g_four_bytes[8];
    double g_eight_bytes[4];
} CacheBlock;

int main(int argc, char* argv[]) {
	string line;
	int lineNum = 0;
	const char* file_path = argv[1];
	ifstream inputfile(file_path); 
	
	file_path = argv[2];
	ofstream outputfile(file_path);
	if (!outputfile.is_open()) {
        std::cerr << "파일을 열 수 없습니다." << std::endl;
        return 1;
    }

    map<std::string, std::string> Blocks; 
	Blocks.clear();
	size_t found;
	size_t Launch_flag (0);
	int size;
	if(inputfile.is_open()){
		while(getline(inputfile, line)) {
			found = line.find(Kernel_start_message);
			if(found != std::string::npos) { //Kernel Launch
				Launch_flag = 1;
					if(!Blocks.empty()){
						//fflush map
						//std::cout << "Kernel dump size : " << Blocks.size() << std::endl;
						for(const auto& pair : Blocks) {
							outputfile<< pair.second << std::endl;
						}
						Blocks.clear();
					}
				//outputfile << line.substr(found) << endl;
			} else {
				if(Launch_flag == 1) {
					// find MREF Size
					found = line.find("Size");
					if(found == std::string::npos) {
						continue;
					} else {
						found += 5;
						size_t start = found;
						while (found < line.length() && !std::isspace(line[found])) {
							found++;
						}
						std::string size_str = line.substr(start, found - start);
						//std::cout << size_str << std::endl;
						size = std::stoi(size_str);
						// collect data(address) from threads(32 per warps)
						for(int i = 0; i < 32; i++) {
							// find data & address & block_address
							found = line.find("Thread", found);
							if(i >= 10)
								found += 9;
							else
								found += 8;

							/*
							std::string data;
							HexString_to_BinaryString(line.substr(found+2,DATA_STRING_SIZE),data);
							data = data.substr(64-(size*8));
							*/
							int data_string_size = (size == 16) ? 32 : 16;
							std::string data = line.substr(found+2,data_string_size);
							found += 2;
							data = data.substr(data_string_size-(size*2));


							size_t flag;
							flag = data.find(",");
							if(flag != std::string::npos){
								std::cout << size << " " << "find error!" << data << " " << lineNum << std::endl;
								break;
							}



							found = line.find(",", found) + 1;
							std::string address;
							/*
							HexString_to_BinaryString(line.substr(found+2,ADDRESS_STRING_SIZE),address);
							std::string block_address = address.substr(0,59);
							std::string block_offset = address.substr(59);
							*/
							std::string checkstring = line.substr(found+2,ADDRESS_STRING_SIZE);
							HexString_to_BinaryString(line.substr(found+2,ADDRESS_STRING_SIZE),address);
							std::string nulladdress (64,'0');
							if(address.compare(nulladdress)!=0) {
								std::string block_address = address.substr(0,59); 
								std::string block_address_extended = '0' + block_address;
								std::string block_address_h;
								BinarySting_to_HexString(block_address_extended, block_address_h);
								int block_offset = std::stoi(address.substr(59),nullptr,2);
								//std::cout << "ThreadIdx" << i << " " << data << " " << block_address_h 
								//			<< "offset : " << block_offset <<  endl;

								//find and insert Cache-line to Map
								if(Blocks.find(block_address_h) != Blocks.end()) {
									Blocks[block_address_h].replace(64-(block_offset*2)-size*2,size*2,data);
								} else {
									std::string extendedData(64,'0');
									if(64-(block_offset*2)-size*2 < 0) {
										std::cout << checkstring << std::endl;
									}
									extendedData.replace(64-(block_offset*2)-size*2,size*2,data);
									Blocks.insert({block_address_h, extendedData});
								}
							}
						}
					}
				} else {
					continue;
				}
			}
			lineNum++;
		}
		if(!Blocks.empty()){
			for(const auto& pair : Blocks) {
				outputfile<< pair.second << std::endl;
			}
		}
		Blocks.clear();

		inputfile.close();
	} else {
		cout << "Unable to open file";
		return 1;
	}
	return 0;
}

void HexString_to_BinaryString(const std::string& HexString, std::string& BinaryString) {
	//std::string binaryString = "";
    for (char c : HexString) {
        int intValue;
        if (std::isdigit(c)) {
            intValue = c - '0';
        } else {
            c = std::toupper(c);
            intValue = c - 'a' + 10;
        }

        std::bitset<4> binaryValue(intValue);
        BinaryString += binaryValue.to_string();
    }
}

void BinarySting_to_HexString(const std::string& binaryString, std::string& HexString) {
    // binary 문자열을 4개씩 묶어서 hex 문자열로 변환
    //std::string HexString;
    for (size_t i = 0; i < binaryString.length(); i += 4) {
        std::string chunk = binaryString.substr(i, 4);
        int decimalValue = std::stoi(chunk, nullptr, 2);
		char c;
		if(decimalValue < 10)
			c = decimalValue + '0';
		else
			c = decimalValue + 'a' - 10;
		HexString += c;
    }
}
