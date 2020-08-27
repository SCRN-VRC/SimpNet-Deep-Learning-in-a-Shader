/*
	The C++ version of SimpNet is the intermediate step from 
	Python -> C++ -> HLSL in my attempt to code a CNN with 
	back propagation in a fragment shader
	
	I coded this to mimic GPU behavior, optimized for the GPU, 
	so this runs slow CPU wise

	- SCRN
*/

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <regex>
#include "json.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;
using json = nlohmann::json;

// Debugging params
#define L1              1
#define L2              2
#define L3              3
#define FC1             4
#define FC2             5
#define FC3             6
#define DEBUG_ALL       7

#define DEBUG_LAYER     FC3
#define DEBUG_BP		0
#define DEBUG_WEIGHTS   0
#define XORTEST         0
#define TRAIN           1

#define WEIGHTS_PATH    "C:\\Users\\Alan\\source\\repos\\SimpNetPython\\WeightsCPP.txt"
#define TRAIN_DIR       "D:\\Storage\\Datasets\\Train\\"
#define TEST_DIR        "D:\\Storage\\Datasets\\Test\\"
#define CATEGORY        "Fruits"

// Helper functions

// Avoid default to_str method
template < typename Type > string to_str(const Type & t)
{
	ostringstream os;
	os << t;
	return os.str();
}

class CNN {
private:
	float lr;
	float alpha;
	float rho;
	float epsilon;

	// forward prop

	// ouputs
	float*** L1s;       // 32 x 32 x 32
	float*** L1a;       // 32 x 32 x 32
	float*** L1Max;     // 16 x 16 x 32
	uint*** L1iMax;     // 16 x 16 x 32

	float*** L2s;       // 14 x 14 x 64
	float*** L2a;       // 14 x 14 x 64
	float*** L2Max;     // 7 x 7 x 64
	uint*** L2iMax;     // 7 x 7 x 64

	float*** L3s;       // 3 x 3 x 128
	float*** L3a;       // 3 x 3 x 128
	float* L3Max;       // 1 x 1 x 128
	uint* L3iMax;       // 1 x 1 x 128

	float* FC1s;        // 1 x 128
	float* FC1a;        // 1 x 128

	float* FC2s;        // 1 x 128
	float* FC2a;        // 1 x 128

	float* FC3s;        // 1 x 12
	float* FC3o;        // 1 x 12

	// kernels and bias
	float**** wL1;      // 3 x 3 x 3 x 32
	float* bL1;         // 1 x 32

	float**** wL2;      // 3 x 3 x 32 x 64
	float* bL2;         // 1 x 64

	float**** wL3;      // 3 x 3 x 64 x 128
	float* bL3;         // 1 x 128

	float** wFC1;       // 128 x 128
	float* bFC1;        // 1 x 128

	float** wFC2;       // 128 x 128
	float* bFC2;        // 1 x 128

	float** wFC3;       // 128 x 12
	float* bFC3;        // 1 x 12

	// backprop
	float** dwFC3;      // 128 x 12
	float** dwFC3_m;    // 128 x 12
	float* dbFC3;       // 1 x 12
	float* dbFC3_m;     // 1 x 12

	float** dwFC2;      // 128 x 128
	float** dwFC2_m;    // 128 x 128
	float* dbFC2;       // 1 x 128
	float* dbFC2_m;     // 1 x 128

	float** dwFC1;      // 128 x 128
	float** dwFC1_m;    // 128 x 128
	float* dbFC1;       // 1 x 128
	float* dbFC1_m;     // 1 x 128

	float* eL3Max;      // 1 x 128
	float*** eL3;       // 3 x 3 x 128
	float*** diL3;		// 5 x 5 x 128
	float**** dwL3;     // 3 x 3 x 64 x 128
	float**** dwL3_m;   // 3 x 3 x 64 x 128
	float* dbL3;        // 1 x 128
	float* dbL3_m;      // 1 x 128

	float*** padL3;     // 9 x 9 x 128
	float*** eL2Max;    // 7 x 7 x 64
	float*** eL2;       // 14 x 14 x 64
	float**** dwL2;     // 3 x 3 x 32 x 64
	float**** dwL2_m;   // 3 x 3 x 32 x 64
	float* dbL2;        // 1 x 64
	float* dbL2_m;      // 1 x 64

	float*** padL2;     // 18 x 18 x 64
	float*** eL1Max;    // 16 x 16 x 32
	float*** eL1;       // 32 x 32 x 32
	float*** diL1;		// 63 x 63 x 32
	float**** dwL1;     // 3 x 3 x 3 x 32
	float**** dwL1_m;   // 3 x 3 x 3 x 32
	float* dbL1;        // 1 x 32
	float* dbL1_m;      // 1 x 32

public:

	// Annoying callocs
	static void** createArray(int i, int j, size_t size)
	{
		void** r = (void**)calloc(i, sizeof(void*));
		for (int x = 0; x < i; x++) {
			r[x] = (void*)calloc(j, size);
		}
		return r;
	}

	static void*** createArray(int i, int j, int k, size_t size)
	{
		void*** r = (void***)calloc(i, sizeof(void*));
		for (int x = 0; x < i; x++) {
			r[x] = (void**)calloc(j, sizeof(void*));
			for (int y = 0; y < j; y++) {
				r[x][y] = (void*)calloc(k, size);
			}
		}
		return r;
	}

	static void**** createArray(int i, int j, int k, int l, size_t size)
	{
		void**** r = (void****)calloc(i, sizeof(void*));
		for (int x = 0; x < i; x++) {
			r[x] = (void***)calloc(j, sizeof(void*));
			for (int y = 0; y < j; y++) {
				r[x][y] = (void**)calloc(k, sizeof(void*));
				for (int z = 0; z < k; z++) {
					r[x][y][z] = (void*)calloc(l, size);
				}
			}
		}
		return r;
	}

	// Annoying calloc frees
	static void freeArray(int i, int j, void** a)
	{
		for (int x = 0; x < i; x++) {
			free(a[x]);
		}
		free(a);
	}

	static void freeArray(int i, int j, int k, void*** a)
	{
		for (int x = 0; x < i; x++) {
			for (int y = 0; y < j; y++) {
				free(a[x][y]);
			}
			free(a[x]);
		}
		free(a);
	}

	static void freeArray(int i, int j, int k, int l, void**** a)
	{
		for (int x = 0; x < i; x++) {
			for (int y = 0; y < j; y++) {
				for (int z = 0; z < k; z++) {
					free(a[x][y][z]);
				}
				free(a[x][y]);
			}
			free(a[x]);
		}
		free(a);
	}

	void load(string path)
	{
		ifstream ifs(path);
		if (!ifs) {
			std::cout << "Failed to open: " << path << std::endl;
			return;
		}
		json jf = json::parse(ifs);

		// L1 weights
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 3; k++) {
					for (int l = 0; l < 32; l++) {
						wL1[i][j][k][l] = jf.at(0).at(i).at(j).at(k).at(l);
					}
				}
			}
		}

		// L1 bias
		for (int i = 0; i < 32; i++) {
			bL1[i] = jf.at(1).at(i);
		}

		// L2 weights
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 32; k++) {
					for (int l = 0; l < 64; l++) {
						wL2[i][j][k][l] = jf.at(2).at(i).at(j).at(k).at(l);
					}
				}
			}
		}

		// L2 bias
		for (int i = 0; i < 64; i++) {
			bL2[i] = jf.at(3).at(i);
		}

		// L3 weights
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 64; k++) {
					for (int l = 0; l < 128; l++) {
						wL3[i][j][k][l] = jf.at(4).at(i).at(j).at(k).at(l);
					}
				}
			}
		}

		// L3 bias
		for (int i = 0; i < 128; i++) {
			bL3[i] = jf.at(5).at(i);
		}

		// FC1 weights
		for (int i = 0; i < 128; i++) {
			for (int j = 0; j < 128; j++) {
				wFC1[i][j] = jf.at(6).at(i).at(j);
			}
		}

		// FC1 bias
		for (int i = 0; i < 128; i++) {
			bFC1[i] = jf.at(7).at(i);
		}

		// FC2 weights
		for (int i = 0; i < 128; i++) {
			for (int j = 0; j < 128; j++) {
				wFC2[i][j] = jf.at(8).at(i).at(j);
			}
		}

		// FC2 bias
		for (int i = 0; i < 128; i++) {
			bFC2[i] = jf.at(9).at(i);
		}

		// FC3 weights
		for (int i = 0; i < 128; i++) {
			for (int j = 0; j < 12; j++) {
				wFC3[i][j] = jf.at(10).at(i).at(j);
			}
		}

		// FC3 bias
		for (int i = 0; i < 12; i++) {
			bFC3[i] = jf.at(11).at(i);
		}

		std::cout << "Weights loaded successfully" << std::endl;
	}

	float afn(float x)
	{
		// ELU
		return x >= 0.f ? x : alpha * (exp(x) - 1.f);
		// RELU
		//return max(0.f, x);
		// LeakyRELU
		//return max(alpha * x, x);
		// Sigmoid
		//return 1.f / (1.f + exp(-x));
	}

	float dfn(float x)
	{
		// ELU
		return x >= 0.f ? 1.f : alpha * exp(x);
		// RELU
		//return x > 0.f ? 1.f : 0.f;
		// LeakyRELU
		//return x > 0.f ? 1.f : alpha;
		// Sigmoid
		//return afn(x) * (1.f - afn(x));
	}

	CNN()
	{
		// Tensorflow defaults for RMSprop
		lr = 0.001f;
		alpha = 1.0f;
		rho = 0.9f;
		epsilon = 1e-07f;

		L1s = (float***)createArray(32, 32, 32, sizeof(float));
		L1a = (float***)createArray(32, 32, 32, sizeof(float));

		L1Max = (float***)createArray(16, 16, 32, sizeof(float));
		L1iMax = (uint***)createArray(16, 16, 32, sizeof(uint));

		L2s = (float***)createArray(14, 14, 64, sizeof(float));
		L2a = (float***)createArray(14, 14, 64, sizeof(float));

		L2Max = (float***)createArray(7, 7, 64, sizeof(float));
		L2iMax = (uint***)createArray(7, 7, 64, sizeof(uint));

		L3s = (float***)createArray(3, 3, 128, sizeof(float));
		L3a = (float***)createArray(3, 3, 128, sizeof(float));

		L3Max = (float*)calloc(128, sizeof(float));
		L3iMax = (uint*)calloc(128, sizeof(uint));

		FC1s = (float*)calloc(128, sizeof(float));
		FC1a = (float*)calloc(128, sizeof(float));
		FC2s = (float*)calloc(128, sizeof(float));
		FC2a = (float*)calloc(128, sizeof(float));

		FC3s = (float*)calloc(12, sizeof(float));
		FC3o = (float*)calloc(12, sizeof(float));

		padL2 = (float***)createArray(18, 18, 64, sizeof(float));
		eL1Max = (float***)createArray(16, 16, 32, sizeof(float));
		eL1 = (float***)createArray(32, 32, 32, sizeof(float));
		diL1 = (float***)createArray(63, 63, 32, sizeof(float));

		wL1 = (float****)createArray(3, 3, 3, 32, sizeof(float));
		dwL1 = (float****)createArray(3, 3, 3, 32, sizeof(float));
		dwL1_m = (float****)createArray(3, 3, 3, 32, sizeof(float));

		random_device rd;  //Will be used to obtain a seed for the random number engine
		mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
#if (DEBUG_WEIGHTS)
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 3; k++) {
					for (int l = 0; l < 32; l++) {
						wL1[i][j][k][l] = i * j * k / (l + 1.0f);
					}
				}
			}
		}
#else
		normal_distribution<float> dis1(0.0f, 1.0f / sqrt(27.0f));
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 3; k++) {
					for (int l = 0; l < 32; l++) {
						wL1[i][j][k][l] = dis1(gen);
					}
				}
			}
		}
#endif

		padL3 = (float***)createArray(9, 9, 128, sizeof(float));
		eL2Max = (float***)createArray(7, 7, 64, sizeof(float));
		eL2 = (float***)createArray(14, 14, 64, sizeof(float));

		wL2 = (float****)createArray(3, 3, 32, 64, sizeof(float));
		dwL2 = (float****)createArray(3, 3, 32, 64, sizeof(float));
		dwL2_m = (float****)createArray(3, 3, 32, 64, sizeof(float));
#if (DEBUG_WEIGHTS)
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 32; k++) {
					for (int l = 0; l < 64; l++) {
						wL2[i][j][k][l] = (i + j + k + l) / 1000.0f;
					}
				}
			}
		}
#else
		normal_distribution<float> dis2(0.0f, 1.0f / sqrt(288.0f));
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 32; k++) {
					for (int l = 0; l < 64; l++) {
						wL2[i][j][k][l] = dis2(gen);
					}
				}
			}
		}
#endif

		eL3Max = (float*)calloc(128, sizeof(float));
		eL3 = (float***)createArray(3, 3, 128, sizeof(float));
		diL3 = (float***)createArray(5, 5, 128, sizeof(float));

		wL3 = (float****)createArray(3, 3, 64, 128, sizeof(float));
		dwL3 = (float****)createArray(3, 3, 64, 128, sizeof(float));
		dwL3_m = (float****)createArray(3, 3, 64, 128, sizeof(float));
#if (DEBUG_WEIGHTS)
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 64; k++) {
					for (int l = 0; l < 128; l++) {
						wL3[i][j][k][l] = (i + j) / float(k + l + 1.0f);
					}
				}
			}
	}
#else
		normal_distribution<float> dis3(0.0f, 1.0f / sqrt(576.0f));
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 64; k++) {
					for (int l = 0; l < 128; l++) {
						wL3[i][j][k][l] = dis3(gen);
					}
				}
			}
		}
#endif

		wFC1 = (float**)createArray(128, 128, sizeof(float));
		wFC2 = (float**)createArray(128, 128, sizeof(float));
		dwFC2 = (float**)createArray(128, 128, sizeof(float));
		dwFC2_m = (float**)createArray(128, 128, sizeof(float));
		dwFC1 = (float**)createArray(128, 128, sizeof(float));
		dwFC1_m = (float**)createArray(128, 128, sizeof(float));
#if (DEBUG_WEIGHTS)
		for (int i = 0; i < 128; i++) {
			for (int j = 0; j < 128; j++) {
				wFC1[i][j] = (i + j) / float(i * j + 100000.0f);
				wFC2[i][j] = (i + j) / float(i * j + 200000.0f);
			}
}
#else
		normal_distribution<float> dis4(0.0f, 1.0f / sqrt(128.0f));
		for (int i = 0; i < 128; i++) {
			for (int j = 0; j < 128; j++) {
				wFC1[i][j] = dis4(gen);
				wFC2[i][j] = dis4(gen);
			}
		}
#endif

		wFC3 = (float**)createArray(128, 12, sizeof(float));
		dwFC3 = (float**)createArray(128, 12, sizeof(float));
		dwFC3_m = (float**)createArray(128, 12, sizeof(float));
#if (DEBUG_WEIGHTS)
		for (int i = 0; i < 128; i++) {
			for (int j = 0; j < 12; j++) {
				wFC3[i][j] = (i + 12 - j) / float(j + 2000.f);
			}
		}
#else
		normal_distribution<float> dis5(0.0f, 1.0f / sqrt(128.0f));
		for (int i = 0; i < 128; i++) {
			for (int j = 0; j < 12; j++) {
				wFC3[i][j] = dis5(gen);
			}
		}
#endif

		bL1 = (float*)calloc(32, sizeof(float));
		dbL1 = (float*)calloc(32, sizeof(float));
		dbL1_m = (float*)calloc(32, sizeof(float));

		bL2 = (float*)calloc(64, sizeof(float));
		dbL2 = (float*)calloc(64, sizeof(float));
		dbL2_m = (float*)calloc(64, sizeof(float));

		bL3 = (float*)calloc(128, sizeof(float));
		dbL3 = (float*)calloc(128, sizeof(float));
		dbL3_m = (float*)calloc(128, sizeof(float));

		bFC1 = (float*)calloc(128, sizeof(float));
		bFC2 = (float*)calloc(128, sizeof(float));
		dbFC2 = (float*)calloc(128, sizeof(float));
		dbFC2_m = (float*)calloc(128, sizeof(float));
		dbFC1 = (float*)calloc(128, sizeof(float));
		dbFC1_m = (float*)calloc(128, sizeof(float));

		bFC3 = (float*)calloc(12, sizeof(float));
		dbFC3 = (float*)calloc(12, sizeof(float));
		dbFC3_m = (float*)calloc(12, sizeof(float));

#if (DEBUG_WEIGHTS)
		String o;

		for (int i = 0; i < 32; i++) {
			bL1[i] = i / 32.0f - 0.5f;
		}

		for (int i = 0; i < 64; i++) {
			bL2[i] = 1.0f - (i / 64.0f) - 0.5f;
		}

		for (int i = 0; i < 128; i++) {
			bL3[i] = (i / 128.0f) - 0.5f;
		}

		for (int i = 0; i < 128; i++) {
			bFC1[i] = i / float(i + 100.0f);
		}

		for (int i = 0; i < 128; i++) {
			bFC2[i] = i / float(i + 1000.0f);
		}

		for (int i = 0; i < 12; i++) {
			bFC3[i] = i / 11.f - 0.5f;
		}

		//o += "\nL1 Weights\n";
		//for (int i = 0; i < 3; i++) {
		//	for (int j = 0; j < 3; j++) {
		//		for (int k = 0; k < 3; k++) {
		//			for (int l = 0; l < 32; l++) {
		//				o += to_str(wL1[i][j][k][l]) + " ";
		//			}
		//			o += "\n";
		//		}
		//	}
		//}

		//o += "\nL2 Weights\n";
		//for (int i = 0; i < 3; i++) {
		//	for (int j = 0; j < 3; j++) {
		//		for (int k = 0; k < 32; k++) {
		//			for (int l = 0; l < 64; l++) {
		//				o += to_str(wL2[i][j][k][l]) + " ";
		//			}
		//			o += "\n";
		//		}
		//	}
		//}

		//o += "\nL3 Weights\n";
		//for (int i = 0; i < 3; i++) {
		//	for (int j = 0; j < 3; j++) {
		//		for (int k = 0; k < 64; k++) {
		//			for (int l = 0; l < 128; l++) {
		//				o += to_str(wL3[i][j][k][l]) + " ";
		//			}
		//			o += "\n";
		//		}
		//	}
		//}

		//o += "\nFC1 Weights\n";
		//for (int i = 0; i < 128; i++) {
		//	for (int j = 0; j < 128; j++) {
		//		o += to_str(wFC1[i][j]) + " ";
		//	}
		//	o += "\n";
		//}

		//o += "\nFC2 Weights\n";
		//for (int i = 0; i < 128; i++) {
		//	for (int j = 0; j < 128; j++) {
		//		o += to_str(wFC2[i][j]) + " ";
		//	}
		//	o += "\n";
		//}

		//o += "\nFC3 Weights\n";
		//for (int i = 0; i < 128; i++) {
		//	for (int j = 0; j < 12; j++) {
		//		o += to_str(wFC3[i][j]) + " ";
		//	}
		//	o += "\n";
		//}

		std::cout << o << std::endl;
#endif
	}

	~CNN()
	{
		freeArray(32, 32, 32, (void***)L1s);
		freeArray(32, 32, 32, (void***)L1a);
		freeArray(16, 16, 32, (void***)L1Max);
		freeArray(16, 16, 32, (void***)L1iMax);
		freeArray(14, 14, 64, (void***)L2s);
		freeArray(14, 14, 64, (void***)L2a);
		freeArray(7, 7, 64, (void***)L2Max);
		freeArray(7, 7, 64, (void***)L2iMax);
		freeArray(3, 3, 128, (void***)L3s);
		freeArray(3, 3, 128, (void***)L3a);
		free(L3Max);
		free(L3iMax);
		free(FC1s);
		free(FC1a);
		free(FC2s);
		free(FC2a);
		free(FC3s);
		free(FC3o);
		freeArray(3, 3, 3, 32, (void****)wL1);
		freeArray(3, 3, 32, 64, (void****)wL2);
		freeArray(3, 3, 64, 128, (void****)wL3);
		freeArray(128, 128, (void**)wFC1);
		freeArray(128, 128, (void**)wFC2);
		freeArray(128, 128, (void**)wFC3);
		free(bL1);
		free(bL2);
		free(bL3);
		free(bFC1);
		free(bFC2);
		free(bFC3);

		freeArray(128, 12, (void**)dwFC3);
		freeArray(128, 12, (void**)dwFC3_m);
		free(dbFC3);
		free(dbFC3_m);
		freeArray(128, 128, (void**)dwFC2);
		freeArray(128, 128, (void**)dwFC2_m);
		free(dbFC2);
		free(dbFC2_m);
		freeArray(128, 128, (void**)dwFC1);
		freeArray(128, 128, (void**)dwFC1_m);
		free(dbFC1);
		free(dbFC1_m);
		free(eL3Max);
		freeArray(3, 3, 128, (void***)eL3);
		freeArray(5, 5, 128, (void***)diL3);
		freeArray(3, 3, 64, 128, (void****)dwL3);
		freeArray(3, 3, 64, 128, (void****)dwL3_m);
		free(dbL3);
		free(dbL3_m);
		freeArray(9, 9, 128, (void***)padL3);
		freeArray(7, 7, 64, (void***)eL2Max);
		freeArray(14, 14, 64, (void***)eL2);
		freeArray(3, 3, 32, 64, (void****)dwL2);
		freeArray(3, 3, 32, 64, (void****)dwL2_m);
		free(dbL2);
		free(dbL2_m);
		freeArray(18, 18, 64, (void***)padL2);
		freeArray(16, 16, 32, (void***)eL1Max);
		freeArray(32, 32, 32, (void***)eL1);
		freeArray(63, 63, 32, (void***)diL1);
		freeArray(3, 3, 3, 32, (void****)dwL1);
		freeArray(3, 3, 3, 32, (void****)dwL1_m);
		free(dbL1);
		free(dbL1_m);
	}

	int forwardProp(float*** image, int classNo, String& o, float* ce)
	{
		// Convolutional layer 1, kernel=3x3, stride=2
		for (int k = 0; k < 32; k++) {
			for (int i = 0; i < 32; i++) {
				for (int j = 0; j < 32; j++) {
					L1s[i][j][k] = 0.0f;

					int i0 = i * 2, i1 = i0 + 1, i2 = i0 + 2;
					int j0 = j * 2, j1 = j0 + 1, j2 = j0 + 2;

					for (int l = 0; l < 3; l++) {
						L1s[i][j][k] +=
							image[i0][j0][l] * wL1[0][0][l][k] +
							image[i0][j1][l] * wL1[0][1][l][k] +
							image[i0][j2][l] * wL1[0][2][l][k] +
							image[i1][j0][l] * wL1[1][0][l][k] +
							image[i1][j1][l] * wL1[1][1][l][k] +
							image[i1][j2][l] * wL1[1][2][l][k] +
							image[i2][j0][l] * wL1[2][0][l][k] +
							image[i2][j1][l] * wL1[2][1][l][k] +
							image[i2][j2][l] * wL1[2][2][l][k];
					}

					L1s[i][j][k] += bL1[k];
					L1a[i][j][k] = afn(L1s[i][j][k]);
				}
			}
		}

#if (DEBUG_LAYER == L1 || DEBUG_LAYER == DEBUG_ALL)
		o += "\nConv Layer 1:\n";
		for (int k = 0; k < 32; k++) {
			for (int i = 0; i < 32; i++) {
				for (int j = 0; j < 32; j++) {
					o += to_str(L1a[i][j][k]) + " ";
				}
				o += "\n";
			}
			o += "\n";
		}
#endif

		// Max pooling layer 1, size=2x2, stride=2
		for (int k = 0; k < 32; k++) {
			for (int i = 0; i < 16; i++) {
				for (int j = 0; j < 16; j++) {
					int i0 = i * 2, i1 = i0 + 1;
					int j0 = j * 2, j1 = j0 + 1;

					float m = L1a[i0][j0][k];
					m = fmaxf(m, L1a[i0][j1][k]);
					m = fmaxf(m, L1a[i1][j0][k]);
					m = fmaxf(m, L1a[i1][j1][k]);
					L1Max[i][j][k] = m;
				}
			}
		}

#if (DEBUG_LAYER == L1 || DEBUG_LAYER == DEBUG_ALL)
		o += "\nMax Layer 1:\n";
		for (int k = 0; k < 32; k++) {
			for (int i = 0; i < 16; i++) {
				for (int j = 0; j < 16; j++) {
					o += to_str(L1Max[i][j][k]) + " ";
				}
				o += "\n";
			}
			o += "\n";
		}
#endif

		// Max pooling layer 1 index
		for (int k = 0; k < 32; k++) {
			for (int i = 0; i < 16; i++) {
				for (int j = 0; j < 16; j++) {
					int i0 = i * 2, i1 = i0 + 1;
					int j0 = j * 2, j1 = j0 + 1;

					float m = L1a[i0][j0][k];
					L1iMax[i][j][k] = i0 * 32 + j0;

					m = fmaxf(m, L1a[i0][j1][k]);
					L1iMax[i][j][k] = (m == L1a[i0][j1][k]) ?
						(i0 * 32 + j1) : L1iMax[i][j][k];

					m = fmaxf(m, L1a[i1][j0][k]);
					L1iMax[i][j][k] = (m == L1a[i1][j0][k]) ?
						(i1 * 32 + j0) : L1iMax[i][j][k];

					m = fmaxf(m, L1a[i1][j1][k]);
					L1iMax[i][j][k] = (m == L1a[i1][j1][k]) ?
						(i1 * 32 + j1) : L1iMax[i][j][k];
				}
			}
		}

#if (DEBUG_LAYER == L1 || DEBUG_LAYER == DEBUG_ALL)
		o += "\nMax Layer 1 Index:\n";
		for (int k = 0; k < 32; k++) {
			for (int i = 0; i < 16; i++) {
				for (int j = 0; j < 16; j++) {
					o += to_str(L1iMax[i][j][k]) + " ";
				}
				o += "\n";
			}
			o += "\n";
		}
#endif
		// Convolutional layer 2, kernel=3x3, stride=1
		for (int k = 0; k < 64; k++) {
			for (int i = 0; i < 14; i++) {
				for (int j = 0; j < 14; j++) {
					L2s[i][j][k] = 0.0f;

					int i0 = i, i1 = i + 1, i2 = i + 2;
					int j0 = j, j1 = j + 1, j2 = j + 2;

					for (int l = 0; l < 32; l++) {
						L2s[i][j][k] +=
							L1Max[i0][j0][l] * wL2[0][0][l][k] +
							L1Max[i0][j1][l] * wL2[0][1][l][k] +
							L1Max[i0][j2][l] * wL2[0][2][l][k] +
							L1Max[i1][j0][l] * wL2[1][0][l][k] +
							L1Max[i1][j1][l] * wL2[1][1][l][k] +
							L1Max[i1][j2][l] * wL2[1][2][l][k] +
							L1Max[i2][j0][l] * wL2[2][0][l][k] +
							L1Max[i2][j1][l] * wL2[2][1][l][k] +
							L1Max[i2][j2][l] * wL2[2][2][l][k];
					}

					L2s[i][j][k] += bL2[k];
					L2a[i][j][k] = afn(L2s[i][j][k]);
				}
			}
		}

#if (DEBUG_LAYER == L2 || DEBUG_LAYER == DEBUG_ALL)
		o += "\nConv Layer 2:\n";
		for (int i = 0; i < 14; i++) {
			for (int j = 0; j < 14; j++) {
				o += to_str(L2s[i][j][0]) + " ";
			}
			o += "\n";
		}
		o += "\n";
#endif

		// Max pooling layer 2, size=2x2, stride=2
		for (int k = 0; k < 64; k++) {
			for (int i = 0; i < 7; i++) {
				for (int j = 0; j < 7; j++) {
					int i0 = i * 2, i1 = i0 + 1;
					int j0 = j * 2, j1 = j0 + 1;

					float m = L2a[i0][j0][k];
					m = fmaxf(m, L2a[i0][j1][k]);
					m = fmaxf(m, L2a[i1][j0][k]);
					m = fmaxf(m, L2a[i1][j1][k]);
					L2Max[i][j][k] = m;
				}
			}
		}

#if (DEBUG_LAYER == L2 || DEBUG_LAYER == DEBUG_ALL)
		o += "\nMax Layer 2:\n";
		for (int i = 0; i < 7; i++) {
			for (int j = 0; j < 7; j++) {
				o += to_str(L2Max[i][j][0]) + " ";
			}
			o += "\n";
		}
		o += "\n";
#endif

		// Max pooling layer 2 index
		for (int k = 0; k < 64; k++) {
			for (int i = 0; i < 7; i++) {
				for (int j = 0; j < 7; j++) {
					int i0 = i * 2, i1 = i0 + 1;
					int j0 = j * 2, j1 = j0 + 1;

					float m = L2a[i0][j0][k];
					L2iMax[i][j][k] = i0 * 14 + j0;
					m = fmaxf(m, L2a[i0][j1][k]);
					L2iMax[i][j][k] = (m == L2a[i0][j1][k]) ?
						(i0 * 14 + j1) : L2iMax[i][j][k];
					m = fmaxf(m, L2a[i1][j0][k]);
					L2iMax[i][j][k] = (m == L2a[i1][j0][k]) ?
						(i1 * 14 + j0) : L2iMax[i][j][k];
					m = fmaxf(m, L2a[i1][j1][k]);
					L2iMax[i][j][k] = (m == L2a[i1][j1][k]) ?
						(i1 * 14 + j1) : L2iMax[i][j][k];
				}
			}
		}

#if (DEBUG_LAYER == L2 || DEBUG_LAYER == DEBUG_ALL)
		o += "\nMax Layer 2 Index:\n";
		for (int i = 0; i < 7; i++) {
			for (int j = 0; j < 7; j++) {
				o += to_str(L2iMax[i][j][0]) + " ";
			}
			o += "\n";
		}
		o += "\n";
#endif

		// Convolutional layer 3, kernel=3x3, stride=2
		for (int k = 0; k < 128; k++) {
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					L3s[i][j][k] = 0.0f;

					int i0 = i * 2, i1 = i0 + 1, i2 = i0 + 2;
					int j0 = j * 2, j1 = j0 + 1, j2 = j0 + 2;

					for (int l = 0; l < 64; l++) {
						L3s[i][j][k] +=
							L2Max[i0][j0][l] * wL3[0][0][l][k] +
							L2Max[i0][j1][l] * wL3[0][1][l][k] +
							L2Max[i0][j2][l] * wL3[0][2][l][k] +
							L2Max[i1][j0][l] * wL3[1][0][l][k] +
							L2Max[i1][j1][l] * wL3[1][1][l][k] +
							L2Max[i1][j2][l] * wL3[1][2][l][k] +
							L2Max[i2][j0][l] * wL3[2][0][l][k] +
							L2Max[i2][j1][l] * wL3[2][1][l][k] +
							L2Max[i2][j2][l] * wL3[2][2][l][k];
					}

					L3s[i][j][k] += bL3[k];
					L3a[i][j][k] = afn(L3s[i][j][k]);
				}
			}
		}

#if (DEBUG_LAYER == L3 || DEBUG_LAYER == DEBUG_ALL)
		o += "\nConv Layer 3:\n";
		for (int k = 0; k < 128; k++) {
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					o += to_str(L3a[i][j][k]) + " ";
				}
				o += "\n";
			}
			o += "\n";
		}
		o += "\n";
#endif

		// Max pooling layer 3, size=3x3, stride=1
		for (int k = 0; k < 128; k++) {
			float m = L3a[0][0][k];
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					m = fmaxf(m, L3a[i][j][k]);
				}
			}
			L3Max[k] = m;
		}

#if (DEBUG_LAYER == L3 || DEBUG_LAYER == DEBUG_ALL)
		o += "\nMax Layer 3:\n";
		for (int k = 0; k < 128; k++) {
			o += to_str(L3Max[k]) + " ";
		}
		o += "\n";
#endif

		// Max pooling layer 3 index
		for (int k = 0; k < 128; k++) {
			float m = L3a[0][0][k];
			L3iMax[k] = 0;
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					m = fmaxf(m, L3a[i][j][k]);
					L3iMax[k] = (m == L3a[i][j][k]) ?
						(i * 3 + j) : L3iMax[k];
				}
			}
		}

#if (DEBUG_LAYER == L3 || DEBUG_LAYER == DEBUG_ALL)
		o += "\nMax Layer 3 Index:\n";
		for (int k = 0; k < 128; k++) {
			o += to_str(L3iMax[k]) + " ";
		}
		o += "\n";
#endif
		// Dense 1
		for (int k = 0; k < 128; k++) {
			FC1s[k] = 0.0;

			for (int l = 0; l < 128; l++) {
				FC1s[k] += L3Max[l] * wFC1[l][k];
			}

			FC1s[k] += bFC1[k];
			FC1a[k] = afn(FC1s[k]);
		}

#if (DEBUG_LAYER == FC1 || DEBUG_LAYER == DEBUG_ALL)
		o += "\nDense Layer 1:\n";
		for (int k = 0; k < 128; k++) {
			o += to_str(FC1a[k]) + " ";
		}
		o += "\n";
#endif
		// Dense 2
		for (int k = 0; k < 128; k++) {
			FC2s[k] = 0.0;

			for (int l = 0; l < 128; l++) {
				FC2s[k] += FC1a[l] * wFC2[l][k];
			}

			FC2s[k] += bFC2[k];
			FC2a[k] = afn(FC2s[k]);
		}

#if (DEBUG_LAYER == FC2 || DEBUG_LAYER == DEBUG_ALL)
		o += "\nDense Layer 2:\n";
		for (int k = 0; k < 128; k++) {
			o += to_str(FC2a[k]) + " ";
		}
		o += "\n";
#endif

		// Output
		for (int i = 0; i < 12; i++) {
			FC3s[i] = 0.0f;

			for (int j = 0; j < 128; j++) {
				FC3s[i] += FC2a[j] * wFC3[j][i];
			}

			FC3s[i] += bFC3[i];
		}

		// Softmax
		for (int i = 0; i < 12; i++) {
			float s = 0.f;
			// Total
			for (int j = 0; j < 12; j++) {
				s += exp(FC3s[j]);
			}
			FC3o[i] = exp(FC3s[i]) / s;
		}

		float expected[12] = { 0.0f };
		expected[classNo] = 1.0f;
		*ce = 0.0f;
		for (int i = 0; i < 12; i++) {
			*ce += expected[i] * log(FC3o[i]);
		}
		*ce = -*ce;

#if (DEBUG_LAYER == FC3 || DEBUG_LAYER == DEBUG_ALL)
		o += "\nsoftmax\n";
		for (int i = 0; i < 12; i++) {
			o += to_str(FC3o[i]);
			o.push_back(' ');
	}
		o.push_back('\n');
#endif
		return (int)(max_element(FC3o, FC3o + 12) - FC3o);
	}

	void backProp(float*** image, int classNo, String &o)
	{
		float expected[12] = { 0.0f };
		expected[classNo] = 1.0f;

		// FC3 bias
		for (int i = 0; i < 12; i++) {
			// Cross Entropy derivative with softmax
			dbFC3[i] = (FC3o[i] - expected[i]);
		}

#if (DEBUG_BP == FC3)
		o += "\nFC3 bias delta:\n";
		for (int i = 0; i < 12; i++) {
			o += to_str(dbFC3[i]) + " ";
		}
		o += "\n";
#endif

		// FC3 gradient
		for (int i = 0; i < 128; i++) {
			for (int j = 0; j < 12; j++) {
				dwFC3[i][j] = dbFC3[j] * FC2a[i];
			}
		}

#if (DEBUG_BP == FC3)
		o += "\nFC3 weight delta:\n";
		for (int i = 0; i < 128; i++) {
			for (int j = 0; j < 12; j++) {
				o += to_str(dwFC3[i][j]) + " ";
			}
			o += "\n";
		}
#endif

		// FC2 bias
		for (int i = 0; i < 128; i++) {
			dbFC2[i] = 0.f;
			for (int j = 0; j < 12; j++) {
				// With respect to w3
				dbFC2[i] += dbFC3[j] * wFC3[i][j];
			}
		}

		// FC2 gradient
		for (int i = 0; i < 128; i++) {
			for (int j = 0; j < 128; j++) {
				// With respect to the activation function of fc2 and the output of previous layer
				dwFC2[i][j] = dbFC2[j] * dfn(FC2s[i]) * FC1a[i];
			}
		}

		// FC1 bias
		for (int i = 0; i < 128; i++) {
			dbFC1[i] = 0.0f;
			for (int j = 0; j < 128; j++) {
				// With respect to activation function of fc2 and w2
				dbFC1[i] += dbFC2[j] * dfn(FC2s[i]) * wFC2[i][j];
			}
		}

		// FC1 gradient
		for (int i = 0; i < 128; i++) {
			for (int j = 0; j < 128; j++) {
				// With respect to activation function of fc1 and the output of previous layer
				dwFC1[i][j] = dbFC1[j] * dfn(FC1s[i]) * L3Max[i];
			}
		}

		// L3 error
		for (int i = 0; i < 128; i++) {
			eL3Max[i] = 0.0f;
			for (int j = 0; j < 128; j++) {
				// Figure out the loss w.r.t weight
				eL3Max[i] += dbFC1[j] * dfn(FC1s[i]) * wFC1[i][j];
			}
		}

		// L3 bias
		for (int i = 0; i < 128; i++) {
			dbL3[i] = eL3Max[i];
		}

		// Undo max pooling 1x1 -> 3x3
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 128; k++) {
					eL3[i][j][k] = L3iMax[k] == (i * 3 + j) ?
						eL3Max[k] * dfn(L3s[j][i][k]) : 0.0f;
				}
			}
		}

#if (DEBUG_BP == L3)
		o += "\neFC3 max pool:\n";
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				o += to_str(eL3[i][j][0]) + " ";
			}
			o += "\n";
		}
#endif

		// Dialate L3 stride=2
		// https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-fb2f2efc4faa
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 5; j++) {
				int i0 = i / 2;
				int j0 = j / 2;
				for (int k = 0; k < 128; k++) {
					diL3[i][j][k] = ((i % 2 == 1) || (j % 2 == 1)) ?
						0.0f : eL3[i0][j0][k];
				}
			}
		}

#if (DEBUG_BP == L3)
		o += "\ndiL3:\n";
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 5; j++) {
				o += to_str(diL3[i][j][0]) + " ";
			}
			o += "\n";
		}
#endif

		// L3 gradient
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 64; k++) {
					for (int l = 0; l < 128; l++) {
						// Convolve 5x5 error over 7x7 input
						float s = 0.0f;
						for (int x = 0; x < 5; x++) {
							for (int y = 0; y < 5; y++) {
								int lx = x + i;
								int ly = y + j;
								s += diL3[x][y][l] * L2Max[lx][ly][k];
							}
						}
						dwL3[i][j][k][l] = s;
					}
				}
			}
		}

#if (DEBUG_BP == L3)
		o += "\ndwL3:\n";
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 64; k++) {
					o += to_str(dwL3[i][j][k][0]) + " ";
				}
				o += "\n";
			}
		}
#endif

		// Pad L3 error
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 9; j++) {
				for (int k = 0; k < 128; k++) {
					// padding = 2
					padL3[i][j][k] = (i < 2 || j < 2 || i > 6 || j > 6) ?
						0.0f : diL3[i - 2][j - 2][k];
				}
			}
		}

		// L2 error
		for (int i = 0; i < 7; i++) {
			for (int j = 0; j < 7; j++) {
				for (int k = 0; k < 64; k++) {
					float s = 0.0f;
					for (int l = 0; l < 128; l++) {
						s += padL3[i + 0][j + 0][l] * wL3[2][2][k][l];
						s += padL3[i + 0][j + 1][l] * wL3[2][1][k][l];
						s += padL3[i + 0][j + 2][l] * wL3[2][0][k][l];
						s += padL3[i + 1][j + 0][l] * wL3[1][2][k][l];
						s += padL3[i + 1][j + 1][l] * wL3[1][1][k][l];
						s += padL3[i + 1][j + 2][l] * wL3[1][0][k][l];
						s += padL3[i + 2][j + 0][l] * wL3[0][2][k][l];
						s += padL3[i + 2][j + 1][l] * wL3[0][1][k][l];
						s += padL3[i + 2][j + 2][l] * wL3[0][0][k][l];
					}
					eL2Max[i][j][k] = s;
				}
			}
		}

		// L2 bias
		for (int i = 0; i < 64; i++) {
			float s = 0.0f;
			for (int x = 0; x < 7; x++) {
				for (int y = 0; y < 7; y++) {
					s += eL2Max[x][y][i];
				}
			}
			dbL2[i] = s;
		}

		// Undo max pooling 7x7 -> 14x14
		for (int i = 0; i < 14; i++) {
			for (int j = 0; j < 14; j++) {
				int i0 = i / 2;
				int j0 = j / 2;
				for (int k = 0; k < 64; k++) {
					eL2[i][j][k] = L2iMax[i0][j0][k] == i * 14 + j ?
						eL2Max[i0][j0][k] * dfn(L2s[j][i][k]) : 0.0f;
				}
			}
		}

		// L2 gradient
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 32; k++) {
					for (int l = 0; l < 64; l++) {
						// Convolve 14x14 error over 16x16 input
						float s = 0.0f;
						for (int x = 0; x < 14; x++) {
							for (int y = 0; y < 14; y++) {
								int lx = x + i;
								int ly = y + j;
								s += eL2[x][y][l] * L1Max[lx][ly][k];
							}
						}
						dwL2[i][j][k][l] = s;
					}
				}
			}
		}

		// Pad L2 error
		for (int i = 0; i < 18; i++) {
			for (int j = 0; j < 18; j++) {
				for (int k = 0; k < 64; k++) {
					padL2[i][j][k] = (i < 2 || j < 2 || i > 15 || j > 15) ?
						0.0f : eL2[i - 2][j - 2][k];
				}
			}
		}

		// L1 error
		for (int i = 0; i < 16; i++) {
			for (int j = 0; j < 16; j++) {
				for (int k = 0; k < 32; k++) {
					float s = 0.0f;
					for (int l = 0; l < 64; l++) {
						s += padL2[i + 0][j + 0][l] * wL2[2][2][k][l];
						s += padL2[i + 0][j + 1][l] * wL2[2][1][k][l];
						s += padL2[i + 0][j + 2][l] * wL2[2][0][k][l];
						s += padL2[i + 1][j + 0][l] * wL2[1][2][k][l];
						s += padL2[i + 1][j + 1][l] * wL2[1][1][k][l];
						s += padL2[i + 1][j + 2][l] * wL2[1][0][k][l];
						s += padL2[i + 2][j + 0][l] * wL2[0][2][k][l];
						s += padL2[i + 2][j + 1][l] * wL2[0][1][k][l];
						s += padL2[i + 2][j + 2][l] * wL2[0][0][k][l];
					}
					eL1Max[i][j][k] = s;
				}
			}
		}

		// L1 bias
		for (int i = 0; i < 32; i++) {
			float s = 0.0f;
			for (int x = 0; x < 16; x++) {
				for (int y = 0; y < 16; y++) {
					s += eL1Max[x][y][i];
				}
			}
			dbL1[i] = s;
		}

		// Undo max pool 16x16 -> 32x32
		for (int i = 0; i < 32; i++) {
			for (int j = 0; j < 32; j++) {
				int i0 = i / 2;
				int j0 = j / 2;
				for (int k = 0; k < 32; k++) {
					eL1[i][j][k] = L1iMax[i0][j0][k] == (i * 32 + j) ?
						eL1Max[i0][j0][k] * dfn(L1s[j][i][k]) : 0.0f;
				}
			}
		}

		// Dialate L1 stride=2
		for (int i = 0; i < 63; i++) {
			for (int j = 0; j < 63; j++) {
				int i0 = i / 2;
				int j0 = j / 2;
				for (int k = 0; k < 32; k++) {
					diL1[i][j][k] = ((i % 2 == 1) || (j % 2 == 1)) ?
						0.0f : eL1[i0][j0][k];
				}
			}
		}

		//for (int i = 0; i < 63; i++) {
		//	for (int j = 0; j < 63; j++) {
		//		if (abs(diL1[i][j][0]) * 10000 > 0) {
		//			cout << i << " " << j << " " << endl;
		//		}
		//	}
		//}

		// L1 gradient
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 3; k++) {
					for (int l = 0; l < 32; l++) {
						// Convolve 63x63 error over 65x65 input
						float s = 0.0f;
						for (int x = 0; x < 63; x++) {
							for (int y = 0; y < 63; y++) {
								int lx = x + i;
								int ly = y + j;
								s += diL1[x][y][l] * image[lx][ly][k];
							}
						}
						dwL1[i][j][k][l] = s;
					}
				}
			}
		}
	}

	inline float momentum(float grad, float vd_m)
	{
		return (rho * vd_m) + ((1.0f - rho) * grad * grad);
	}

	// Using RMSprop
	void update(String &o)
	{
		// FC3 bias
		for (int i = 0; i < 12; i++) {
			float vdb = momentum(dbFC3[i], dbFC3_m[i]);
			// Save history
			dbFC3_m[i] = vdb;
			bFC3[i] -= lr * (dbFC3[i] / (sqrt(vdb) + epsilon));
		}

		// FC3 weights
		for (int i = 0; i < 128; i++) {
			for (int j = 0; j < 12; j++) {
				float vdw = momentum(dwFC3[i][j], dwFC3_m[i][j]);
				dwFC3_m[i][j] = vdw;
				wFC3[i][j] -= lr * (dwFC3[i][j] / (sqrt(vdw) + epsilon));
			}
		}

		// FC2 bias
		for (int i = 0; i < 128; i++) {
			float vdb = momentum(dbFC2[i], dbFC2_m[i]);
			dbFC2_m[i] = vdb;
			bFC2[i] -= lr * (dbFC2[i] / (sqrt(vdb) + epsilon));
		}

		// FC2 weights
		for (int i = 0; i < 128; i++) {
			for (int j = 0; j < 128; j++) {
				float vdw = momentum(dwFC2[i][j], dwFC2_m[i][j]);
				dwFC2_m[i][j] = vdw;
				wFC2[i][j] -= lr * (dwFC2[i][j] / (sqrt(vdw) + epsilon));
			}
		}

		// FC1 bias
		for (int i = 0; i < 128; i++) {
			float vdb = momentum(dbFC1[i], dbFC1_m[i]);
			dbFC1_m[i] = vdb;
			bFC1[i] -= lr * (dbFC1[i] / (sqrt(vdb) + epsilon));
		}

		// FC1 weights
		for (int i = 0; i < 128; i++) {
			for (int j = 0; j < 128; j++) {
				float vdw = momentum(dwFC1[i][j], dwFC1_m[i][j]);
				dwFC1_m[i][j] = vdw;
				wFC1[i][j] -= lr * (dwFC1[i][j] / (sqrt(vdw) + epsilon));
			}
		}

		// L3 bias
		for (int i = 0; i < 128; i++) {
			float vdb = momentum(dbL3[i], dbL3_m[i]);
			dbL3_m[i] = vdb;
			bL3[i] -= lr * (dbL3[i] / (sqrt(vdb) + epsilon));
		}

		// L3 weights
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 64; k++) {
					for (int l = 0; l < 128; l++) {
						float vdw = momentum(dwL3[i][j][k][l], dwL3_m[i][j][k][l]);
						dwL3_m[i][j][k][l] = vdw;
						wL3[i][j][k][l] -= lr * (dwL3[i][j][k][l] / (sqrt(vdw) + epsilon));
					}
				}
			}
		}

		// L2 bias
		for (int i = 0; i < 64; i++) {
			float vdb = momentum(dbL2[i], dbL2_m[i]);
			dbL2_m[i] = vdb;
			bL2[i] -= lr * (dbL2[i] / (sqrt(vdb) + epsilon));
		}

		// L2 weights
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 32; k++) {
					for (int l = 0; l < 64; l++) {
						float vdw = momentum(dwL2[i][j][k][l], dwL2_m[i][j][k][l]);
						dwL2_m[i][j][k][l] = vdw;
						wL2[i][j][k][l] -= lr * (dwL2[i][j][k][l] / (sqrt(vdw) + epsilon));
					}
				}
			}
		}

		// L1 bias
		for (int i = 0; i < 32; i++) {
			float vdb = momentum(dbL1[i], dbL1_m[i]);
			dbL1_m[i] = vdb;
			bL1[i] -= lr * (dbL1[i] / (sqrt(vdb) + epsilon));
		}

		// L1 weights
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 3; k++) {
					for (int l = 0; l < 32; l++) {
						float vdw = momentum(dwL1[i][j][k][l], dwL1_m[i][j][k][l]);
						dwL1_m[i][j][k][l] = vdw;
						wL1[i][j][k][l] -= lr * (dwL1[i][j][k][l] / (sqrt(vdw) + epsilon));
					}
				}
			}
		}
	}
};

int main()
{
	CNN testCNN;
	const int imageSize[2] = { 65, 65 };

#if (!XORTEST)
	vector<String> fn;
	string dir = TRAIN_DIR;
	dir.append(CATEGORY);
	dir.append("\\*.*");
	cv::glob(dir, fn, true);

	vector<Mat> images;
	vector<int> img_class;
	size_t count = fn.size();

	unordered_map<String, int> all_classes;
	string rg = CATEGORY;
	rg.append("\\\\([A-Z]\\w+)");
	regex rgx(rg);
	smatch matches;

	for (size_t i = 0; i < count; i++) {
		images.push_back(imread(fn[i]));
		// Find the class in the string
		regex_search(fn[i], matches, rgx);
		// Add the class string as an index map
		pair<String, int> cur_class(matches[1], all_classes.size());
		all_classes.insert(cur_class);
		// Save class table index
		img_class.push_back(all_classes.at(matches[1]));
	}

	float*** floatRBG = (float***)CNN::createArray(imageSize[0], imageSize[1], 3, sizeof(float));

#if (TRAIN)
	// epoch
	for (int e = 0; e < 10; e++) {
		float tce = 0.0f;
		int co = 0;
		// Shuffle
		unsigned seed = chrono::system_clock::now().time_since_epoch().count();
		shuffle(images.begin(), images.end(), default_random_engine(seed));
		shuffle(img_class.begin(), img_class.end(), default_random_engine(seed));

		for (size_t ll = 0; ll < count; ll++) {
			String o;
			Mat img = images[ll];

			for (int i = 0; i < imageSize[0]; i++) {
				for (int j = 0; j < imageSize[1]; j++) {
					for (int k = 0; k < 3; k++) {
						// Flip BGR to RGB
						floatRBG[i][j][k] = img.at<Vec3b>(i, j)[2 - k] / 255.0f;
					}
				}
			}

			float ce = 0.0f;
			int classOut = testCNN.forwardProp(floatRBG, img_class[ll], o, &ce);
			// Accuracy
			co = img_class[ll] == classOut ? co + 1 : co;
			// Loss
			tce += ce;

			o += "training " + to_str(ll) + " expected: " + to_str(img_class[ll]) +
				" was " + to_str(classOut);
			o += "\naccuracy: " + to_str(co / float(ll + 1)) +
				" loss: " + to_str(tce / float(ll + 1));
			testCNN.backProp(floatRBG, img_class[ll], o);
			testCNN.update(o);

			std::cout << o << std::endl;
		}
	}
#else
	testCNN.load(WEIGHTS_PATH);
#endif

	fn.clear();
	dir = TEST_DIR;
	dir.append(CATEGORY);
	dir.append("\\*.*");
	cv::glob(dir, fn, true);

	images.clear();
	img_class.clear();
	count = fn.size();

	for (size_t i = 0; i < count; i++) {
		images.push_back(imread(fn[i]));
		regex_search(fn[i], matches, rgx);
		img_class.push_back(all_classes.at(matches[1]));
	}

	float tce = 0.0f;
	int co = 0;
	for (size_t ll = 0; ll < count; ll++) {
		String o;
		Mat img = images[ll];

		for (int i = 0; i < imageSize[0]; i++) {
			for (int j = 0; j < imageSize[1]; j++) {
				for (int k = 0; k < 3; k++) {
					// Flip BGR to RGB
					floatRBG[i][j][k] = img.at<Vec3b>(i, j)[2 - k] / 255.0f;
				}
			}
		}

		float ce = 0.0f;
		int classOut = testCNN.forwardProp(floatRBG, img_class[ll], o, &ce);
		co = (classOut == img_class[ll]) ? co + 1 : co;
		tce += ce;
		o += "testing " + to_str(ll) + " expected: " + to_str(img_class[ll]) +
			" was " + to_str(classOut);
		o += "\naccuracy: " + to_str(co / float(ll + 1)) +
			" loss: " + to_str(tce / float(ll + 1));

		std::cout << o << std::endl;
	}

	CNN::freeArray(65, 65, 3, (void***)floatRBG);

#else
	// Test with XOR
	random_device rd;
	default_random_engine gen{ rd() };
	uniform_real_distribution<float> dis0(0.0f, 64.0f);

	const int c = 700;
	float**** input = (float****)CNN::createArray(c, imageSize[0], imageSize[1], 3, sizeof(float));
	int* inClass = (int*)calloc(c, sizeof(int));

	// Setup input
	for (int k = 0; k < c; k++) {
		int r0 = (int)floor(dis0(gen));
		int r1 = (int)floor(dis0(gen));
		for (int i = 0; i < imageSize[0]; i++) {
			for (int j = 0; j < imageSize[1]; j++) {
				input[k][i][j][0] = (i == r0 && j == r1) ? 1.0f : 0.0f;
				input[k][i][j][1] = input[k][i][j][0];
				input[k][i][j][2] = input[k][i][j][0];
			}
		}
		// XOR function
		inClass[k] = ((r0 < (imageSize[0] * 0.5f)) == (r1 < (imageSize[1] * 0.5f))) ? 0 : 1;
	}

	float tce = 0.0f;
	int co = 0;
	for (int i = 0; i < c; i++) {
		String o;
		float ce = 0.0f;
		int classOut = testCNN.forwardProp(input[i], inClass[i], o, &ce);
		// Accuracy
		co = inClass[i] == classOut ? co + 1 : co;
		// Loss
		tce += ce;
		o += "training " + to_str(i) + " expected: " + to_str(inClass[i]) +
			" was " + to_str(classOut);
		o += "\naccuracy: " + to_str(co / float(i + 1)) +
			" loss: " + to_str(tce / float(i + 1));
		testCNN.backProp(input[i], inClass[i], o);
		testCNN.update(o);
		std::cout << o << std::endl;
	}

	const int t = 100;
	float**** test = (float****)CNN::createArray(t, 65, 65, 3, sizeof(float));
	int* testClass = (int*)calloc(t, sizeof(int));

	// Setup input
	for (int k = 0; k < t; k++) {
		int r0 = (int)floor(dis0(gen));
		int r1 = (int)floor(dis0(gen));
		for (int i = 0; i < 65; i++) {
			for (int j = 0; j < 65; j++) {
				test[k][i][j][0] = (i == r0 && j == r1) ? 1.0f : 0.0f;
				test[k][i][j][1] = test[k][i][j][0];
				test[k][i][j][2] = test[k][i][j][0];
			}
		}
		// XOR function
		testClass[k] = ((r0 < (imageSize[0] * 0.5f)) == (r1 < (imageSize[1] * 0.5f))) ? 0 : 1;
	}

	tce = 0.0f;
	co = 0;
	for (int i = 0; i < t; i++) {
		String o;
		float ce = 0.0f;
		int classOut = testCNN.forwardProp(test[i], testClass[i], o, &ce);
		// Accuracy
		co = testClass[i] == classOut ? co + 1 : co;
		// Loss
		tce += ce;
		o += "testing " + to_str(i) + " expected: " + to_str(testClass[i]) +
			" was " + to_str(classOut);
		o += "\naccuracy: " + to_str(co / float(i + 1)) +
			" loss: " + to_str(tce / float(i + 1));
		std::cout << o << std::endl;
	}

	CNN::freeArray(c, imageSize[0], imageSize[1], 3, (void****)input);
	CNN::freeArray(t, imageSize[0], imageSize[1], 3, (void****)test);
	free(inClass);
	free(testClass);

#endif

	system("pause");

	return 0;
}