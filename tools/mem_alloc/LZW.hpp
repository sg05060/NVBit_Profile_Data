#include <string>
#include <list>
#include <iostream>
#include <type_traits>
#include <math.h>
#include <map>

using namespace std;

class LZW {
    private:
		size_t symbol_bits; // Multiples of 4
		size_t codeword_bits;
		size_t encoding_table_size;
		size_t decoding_table_size;
		size_t read_cnt;
		double compression_ratio;
		
		map<std::string, unsigned int> encoding_table;
		map<unsigned int, std::string> decoding_table;
		std::list<std::string> hexalist = {"0","1","2","3","4","5","6","7","8","9","a","b","c","d","e","f"};
	public:
	double getCPR() {
		return compression_ratio;
	}
	void printEncodingTable() {
		for(const auto& pair : encoding_table) {
			std::cout << "["<< pair.first << ":"<< pair.second << "]" << std::endl;
		}
	}
	void printDecodingTable() {
		for(const auto& pair : decoding_table) {
			std::cout << "["<< pair.first << ":"<< pair.second << "]" << std::endl;
		}
	}
	LZW(size_t _symbol_bits, size_t _codeword_bits) {
		symbol_bits = _symbol_bits;
		codeword_bits = _codeword_bits;
		encoding_table_size = 0;
		decoding_table_size = 0;
		read_cnt = _symbol_bits / 4;
	}

	void init_table() {
		std::list<std::string> tmp_str;
		std::list<std::string> update_str;
		for (int i = 0; i < (symbol_bits / 4); i++) {
			if(i == 0) {
				for(auto &hexa : hexalist) {
					update_str.push_back(hexa);
				}
			} else {
				for(auto &update : update_str) {
					for(auto &hexa : hexalist) {
						tmp_str.push_back(update+hexa);
					}
				}
				update_str.clear();
				for(auto &tmp : tmp_str) {
					update_str.push_back(tmp);
				}
				tmp_str.clear();
			}
		}
		for(auto &codeword : update_str) {
			encoding_table[codeword] = encoding_table_size++;
			decoding_table[decoding_table_size++] = codeword;
		}
		
	}

	std::list<unsigned int> compress(std::string input) {
		std::list<unsigned int> compressed;
		std::string w = "";
		
		/*
		for(size_t i = 0; i < input.size()-1; i+=2) {
			std::string c;
			c += input[i];
			c += input[i+1];
			std::string wc = w + c;

			if (encoding_table.find(wc) != encoding_table.end()) { // Find -> 이어나가기, n-gram확장
				w = wc;
			}
			else { // Not found, w는 이전것, wc는 이전거에 지금문자 더한거, 
				compressed.push_back(encoding_table[w]);
				w = c;
				if(encoding_table_size < pow(2,codeword_bits))
					encoding_table[wc] = encoding_table_size++;
			}
		}
		*/
		for(size_t i = 0; i < input.size()- read_cnt + 1; i+=read_cnt) {
			std::string c;
			for(size_t j = 0; j < read_cnt; j++) {
				c += input[i+j];
			}
			std::string wc = w + c;

			if (encoding_table.find(wc) != encoding_table.end()) { // Find -> 이어나가기, n-gram확장
				w = wc;
			}
			else { // Not found, w는 이전것, wc는 이전거에 지금문자 더한거, 
				compressed.push_back(encoding_table[w]);
				w = c;
				if(encoding_table_size < pow(2,codeword_bits))
					encoding_table[wc] = encoding_table_size++;
			}
		}

		if (w != "") {
			compressed.push_back(encoding_table[w]);
		}

		// Compression Fishished
		compression_ratio = (double)(input.length() * 4 ) / (double)(compressed.size() * codeword_bits);
		return compressed;
	}

	std::string decompress(std::list<unsigned int> compressed) {
	

		std::string v = decoding_table[compressed.front()];
		compressed.pop_front();

		std::string decompressed = "";
		decompressed += v;
		std::string pv = v;
		for (auto& c : compressed) {
			if (decoding_table.find(c) == decoding_table.end()) {
				v = pv;
				for(size_t i = 0; i < read_cnt; i++) 
					v += pv[i];
			}
			else {
				v = decoding_table[c];
			}

			decompressed += v;

			/*
			if(decoding_table_size < pow(2,codeword_bits))
				decoding_table[decoding_table_size++] = pv + v[0]+v[1];
			*/
			if(decoding_table_size < pow(2,codeword_bits)) {
				std::string tmp;
				tmp += pv;
				for(size_t i = 0; i < read_cnt; i++) 
					tmp += v[i];
				decoding_table[decoding_table_size++] = tmp;
			}
			
			pv = v;
		}

		//Decompression Finished

		return decompressed;
	}
};