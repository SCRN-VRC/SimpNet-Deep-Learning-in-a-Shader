#include <math.h>
#include <stdlib.h>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <string>
#include <chrono>
#include <unordered_map>
#include <regex>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

// Avoid default to_string method
template < typename Type > string to_str(const Type & t)
{
	ostringstream os;
	os << t;
	return os.str();
}

inline int clamp(int x, int y, int z)
{
	return x < y ? y : x > z ? z : x;
}

inline float actFn(float x) {
	// Sigmoid
	//return 1.0f / (1.0f + exp(-x));
	// ELU
	return x >= 0.0f ? x : (0.15f * (exp(x) - 1.0f));
	// RELU
	//return fmaxf(0.0f, x);
}
inline float dactFn(float x) {
	// Sigmoid
	//return x * (1.0f - x);
	// ELU
	return x >= 0.0f ? 1.0f : exp(x) * 0.15f;
	// RELU
	//return (x < 0.f ? 0.f : 1.0f);
}

// Learning rate
float lr = 0.3f;
// Bias learning rate
float lrb = 0.1f;

float trainImg[600][65][65][3] = { 0.0f };
int trainClass[600] = { 0 };

float testImg[100][65][65][3] = { 0.0f };
int testClass[100] = { 0 };

// L1
float kern1[3][3][3][32]; // 3x3x3x32
float bias1[32];
float convL1s[32][32][32] = { 0.0f }; // 32x32x32
float convL1a[32][32][32] = { 0.0f }; // 32x32x32
float maxL1[16][16][32] = { 0.0f }; // 16x16x32
int imaxL1[16][16][32] = { 0 }; // 16x16x32
// L2
float kern2[3][3][32][64]; // 3x3x32x64
float bias2[64];
float convL2s[14][14][64] = { 0.0f }; // 14x14x64
float convL2a[14][14][64] = { 0.0f }; // 14x14x64
float maxL2[7][7][64] = { 0.0f }; // 7x7x64
int imaxL2[7][7][64] = { 0 }; // 7x7x64
// L3
float kern3[3][3][64][128]; // 3x3x64x128
float bias3[128];
float convL3s[4][4][128] = { 0.0f }; // 4x4x128
float convL3a[4][4][128] = { 0.0f }; // 4x4x128
float maxL3[2][2][128] = { 0.0f }; // 2x2x128
int imaxL3[2][2][128] = { 0 }; // 2x2x128
// L4
float w1[2][2][128][128]; // 512x128
float biasw1[128];
float fc1s[128] = { 0.0f }; // 1x1x128
float fc1a[128] = { 0.0f }; // 1x1x128
// L5
float w2[128][128]; // 512x128
float biasw2[128];
float fc2s[128] = { 0.0f }; // 1x1x128
float fc2a[128] = { 0.0f }; // 1x1x128
// L6
float w3[128][12]; // 128x12
float biasw3[12];
float softout[12] = { 0.0f }; // 1x1x12
float softout2[12] = { 0.0f }; // 1x1x12

// Backprop
float dw3[128][12] = { 0.0f }; // Fully connected output to fc2a
float dbiasw3[12] = { 0.0f }; // Bias for w3
float dw2[128][128] = { 0.0f }; // Fully connected fc2a to fc1a
float dbiasw2[128] = { 0.0f }; // Bias for w2
float dw1[2][2][128][128] = { 0.0f }; // Fully connected fc1a to L3
float dbiasw1[128] = { 0.0f }; // Bias for w1

float emaxL3[2][2][128] = { 0.0f }; // L3 loss input
float dbias3[128] = { 0.0f }; // L3 kernel bias
float econvL3[4][4][128] = { 0.0f }; // Undo max pool for L3
float diconvL3[7][7][128] = { 0.0f }; // L3 dialation
float dkern3[3][3][64][128] = { 0.0f }; // L3 kernel gradient

float emaxL2[7][7][64] = { 0.0f }; // L2 loss input
float dbias2[64] = { 0.0f }; // L2 kernel bias
float econvL2[14][14][64] = { 0.0f }; // Undo max pool for L2
float dkern2[3][3][32][64] = { 0.0f }; // L2 kernel gradient

float pconvL2[18][18][64] = { 0.0f }; // Padded L2 to calculate L1
float emaxL1[16][16][32] = { 0.0f }; // L1 loss input
float dbias1[32] = { 0.0f }; // L1 kernel bias
float econvL1[32][32][32] = { 0.0f }; // Undo max pool for L1
float diconvL1[63][63][32] = { 0.0f }; // L1 dialation
float dkern1[3][3][3][32] = { 0.0f }; // L1 kernel gradient

int doForwardProp(size_t ll, float image[65][65][3], int img_class, String& out)
{
	// Convolutional layer 1, kernel=3x3, stride=2
	for (int k = 0; k < 32; k++) {
		for (int i = 0; i < 32; i++) {
			for (int j = 0; j < 32; j++) {
				convL1s[i][j][k] = 0.0f;

				int i0 = i * 2, i1 = i0 + 1, i2 = i0 + 2;
				int j0 = j * 2, j1 = j0 + 1, j2 = j0 + 2;

				//// Sample image
				for (int l = 0; l < 3; l++) {
					convL1s[i][j][k] +=
						image[i0][j0][l] * kern1[0][0][l][k] +
						image[i0][j1][l] * kern1[0][1][l][k] +
						image[i0][j2][l] * kern1[0][2][l][k] +
						image[i1][j0][l] * kern1[1][0][l][k] +
						image[i1][j1][l] * kern1[1][1][l][k] +
						image[i1][j2][l] * kern1[1][2][l][k] +
						image[i2][j0][l] * kern1[2][0][l][k] +
						image[i2][j1][l] * kern1[2][1][l][k] +
						image[i2][j2][l] * kern1[2][2][l][k];

					//convL1s[i][j][k] +=
					//	image.at<Vec3b>(i0, j0)[l] / 255.0f * kern1[0][0][l][k] +
					//	image.at<Vec3b>(i0, j1)[l] / 255.0f * kern1[0][1][l][k] +
					//	image.at<Vec3b>(i0, j2)[l] / 255.0f * kern1[0][2][l][k] +
					//	image.at<Vec3b>(i1, j0)[l] / 255.0f * kern1[1][0][l][k] +
					//	image.at<Vec3b>(i1, j1)[l] / 255.0f * kern1[1][1][l][k] +
					//	image.at<Vec3b>(i1, j2)[l] / 255.0f * kern1[1][2][l][k] +
					//	image.at<Vec3b>(i2, j0)[l] / 255.0f * kern1[2][0][l][k] +
					//	image.at<Vec3b>(i2, j1)[l] / 255.0f * kern1[2][1][l][k] +
					//	image.at<Vec3b>(i2, j2)[l] / 255.0f * kern1[2][2][l][k];
				}
				// Bias
				convL1s[i][j][k] += bias1[k];

				// Activation
				convL1a[i][j][k] = actFn(convL1s[i][j][k]);
			}
		}
	}

	// Max pooling layer 1, size=2x2, stride=2
	for (int k = 0; k < 32; k++) {
		for (int i = 0; i < 16; i++) {
			for (int j = 0; j < 16; j++) {
				int i0 = i * 2, i1 = i0 + 1;
				int j0 = j * 2, j1 = j0 + 1;

				float m = convL1a[i0][j0][k];
				m = fmaxf(m, convL1a[i0][j1][k]);
				m = fmaxf(m, convL1a[i1][j0][k]);
				m = fmaxf(m, convL1a[i1][j1][k]);
				maxL1[i][j][k] = m;
			}
		}
	}

	// Max pooling layer 1 index
	for (int k = 0; k < 32; k++) {
		for (int i = 0; i < 16; i++) {
			for (int j = 0; j < 16; j++) {
				int i0 = i * 2, i1 = i0 + 1;
				int j0 = j * 2, j1 = j0 + 1;

				float m = convL1a[i0][j0][k];
				imaxL1[i][j][k] = i0 * 32 + j0;

				m = fmaxf(m, convL1a[i0][j1][k]);
				imaxL1[i][j][k] = (m == convL1a[i0][j1][k]) ?
					(i0 * 32 + j1) : imaxL1[i][j][k];

				m = fmaxf(m, convL1a[i1][j0][k]);
				imaxL1[i][j][k] = (m == convL1a[i1][j0][k]) ?
					(i1 * 32 + j0) : imaxL1[i][j][k];

				m = fmaxf(m, convL1a[i1][j1][k]);
				imaxL1[i][j][k] = (m == convL1a[i1][j1][k]) ?
					(i1 * 32 + j1) : imaxL1[i][j][k];
			}
		}
	}

	// Convolutional layer 2, kernel=3x3, stride=1
	for (int k = 0; k < 64; k++) {
		for (int i = 0; i < 14; i++) {
			for (int j = 0; j < 14; j++) {
				convL2s[i][j][k] = 0.0f;

				int i0 = i, i1 = i + 1, i2 = i + 2;
				int j0 = j, j1 = j + 1, j2 = j + 2;

				for (int l = 0; l < 32; l++) {
					convL2s[i][j][k] +=
						maxL1[i0][j0][l] * kern2[0][0][l][k] +
						maxL1[i0][j1][l] * kern2[0][1][l][k] +
						maxL1[i0][j2][l] * kern2[0][2][l][k] +
						maxL1[i1][j0][l] * kern2[1][0][l][k] +
						maxL1[i1][j1][l] * kern2[1][1][l][k] +
						maxL1[i1][j2][l] * kern2[1][2][l][k] +
						maxL1[i2][j0][l] * kern2[2][0][l][k] +
						maxL1[i2][j1][l] * kern2[2][1][l][k] +
						maxL1[i2][j2][l] * kern2[2][2][l][k];
				}

				// Bias
				convL2s[i][j][k] += bias2[k];

				// Activation
				convL2a[i][j][k] = actFn(convL2s[i][j][k]);
			}
		}
	}

	// Max pooling layer 2, size=2x2, stride=2
	for (int k = 0; k < 64; k++) {
		for (int i = 0; i < 7; i++) {
			for (int j = 0; j < 7; j++) {
				int i0 = i * 2, i1 = i0 + 1;
				int j0 = j * 2, j1 = j0 + 1;

				float m = convL2a[i0][j0][k];
				m = fmaxf(m, convL2a[i0][j1][k]);
				m = fmaxf(m, convL2a[i1][j0][k]);
				m = fmaxf(m, convL2a[i1][j1][k]);
				maxL2[i][j][k] = m;
			}
		}
	}

	// Max pooling layer 2 index
	for (int k = 0; k < 64; k++) {
		for (int i = 0; i < 7; i++) {
			for (int j = 0; j < 7; j++) {
				int i0 = i * 2, i1 = i0 + 1;
				int j0 = j * 2, j1 = j0 + 1;

				float m = convL2a[i0][j0][k];
				imaxL2[i][j][k] = i0 * 14 + j0;
				m = fmaxf(m, convL2a[i0][j1][k]);
				imaxL2[i][j][k] = (m == convL2a[i0][j1][k]) ?
					(i0 * 14 + j1) : imaxL2[i][j][k];
				m = fmaxf(m, convL2a[i1][j0][k]);
				imaxL2[i][j][k] = (m == convL2a[i1][j0][k]) ?
					(i1 * 14 + j0) : imaxL2[i][j][k];
				m = fmaxf(m, convL2a[i1][j1][k]);
				imaxL2[i][j][k] = (m == convL2a[i1][j1][k]) ?
					(i1 * 14 + j1) : imaxL2[i][j][k];
			}
		}
	}

	// Convolutional layer 3, kernel=3x3, pad=1, stride=2
	for (int k = 0; k < 128; k++) {
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				convL3s[i][j][k] = { 0.0f };

				int i0 = i * 2, i1 = i0 - 1, i2 = i0 + 1;
				int j0 = j * 2, j1 = j0 - 1, j2 = j0 + 1;

				// Padding
				bool bi1 = i1 < 0, bj1 = j1 < 0, bi2 = i2 > 6, bj2 = j2 > 6;
				bool b02 = bi1 || bj1, b03 = bi1 || bj2, b04 = bi2 || bj1, b05 = bi2 || bj2;

				for (int l = 0; l < 64; l++) {
					convL3s[i][j][k] +=
						(b02 ? 0.0f : maxL2[i1][j1][l] * kern3[0][0][l][k]) +
						(bi1 ? 0.0f : maxL2[i1][j0][l] * kern3[0][1][l][k]) +
						(b03 ? 0.0f : maxL2[i1][j2][l] * kern3[0][2][l][k]) +
						(bj1 ? 0.0f : maxL2[i0][j1][l] * kern3[1][0][l][k]) +
						maxL2[i0][j0][l] * kern3[1][1][l][k] +
						(bj2 ? 0.0f : maxL2[i0][j2][l] * kern3[1][2][l][k]) +
						(b04 ? 0.0f : maxL2[i2][j1][l] * kern3[2][0][l][k]) +
						(bi2 ? 0.0f : maxL2[i2][j0][l] * kern3[2][1][l][k]) +
						(b05 ? 0.0f : maxL2[i2][j2][l] * kern3[2][2][l][k]);
				}
				// Bias
				convL3s[i][j][k] += bias3[k];

				// Activation
				convL3a[i][j][k] = actFn(convL3s[i][j][k]);
			}
		}
	}

	// Max pooling layer 3, size=2x2, stride=2
	for (int k = 0; k < 128; k++) {
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				int i0 = i * 2, i1 = i0 + 1;
				int j0 = j * 2, j1 = j0 + 1;

				float m = convL3a[i0][j0][k];
				m = fmaxf(m, convL3a[i0][j1][k]);
				m = fmaxf(m, convL3a[i1][j0][k]);
				m = fmaxf(m, convL3a[i1][j1][k]);
				maxL3[i][j][k] = m;
			}
		}
	}

	// Max pooling layer 3 index
	for (int k = 0; k < 128; k++) {
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				int i0 = i * 2, i1 = i0 + 1;
				int j0 = j * 2, j1 = j0 + 1;

				float m = convL3a[i0][j0][k];
				imaxL3[i][j][k] = i0 * 4 + j0;
				m = fmaxf(m, convL3a[i0][j1][k]);
				imaxL3[i][j][k] = (m == convL3a[i0][j1][k]) ?
					(i0 * 4 + j1) : imaxL3[i][j][k];
				m = fmaxf(m, convL3a[i1][j0][k]);
				imaxL3[i][j][k] = (m == convL3a[i1][j0][k]) ?
					(i1 * 4 + j0) : imaxL3[i][j][k];
				m = fmaxf(m, convL3a[i1][j1][k]);
				imaxL3[i][j][k] = (m == convL3a[i1][j1][k]) ?
					(i1 * 4 + j1) : imaxL3[i][j][k];
			}
		}
	}

	// FC1
	for (int i = 0; i < 128; i++) {
		fc1s[i] = 0.0f;
		// Summation
		for (int k = 0; k < 2; k++) {
			for (int l = 0; l < 2; l++) {
				for (int j = 0; j < 128; j++) {
					fc1s[i] += maxL3[k][l][j] * w1[k][l][j][i];
				}
			}
		}
		// Bias
		fc1s[i] += biasw1[i];
	}

	// fc1a
	for (int i = 0; i < 128; i++) {
		//Activation
		fc1a[i] = actFn(fc1s[i]);
	}

	// fc2s
	for (int i = 0; i < 128; i++) {
		fc2s[i] = 0.0f;

		// Summation
		for (int j = 0; j < 128; j++) {
			fc2s[i] += fc1a[j] * w2[i][j];
		}
		// Bias
		fc2s[i] += biasw2[i];
	}

	// fc2a
	for (int i = 0; i < 128; i++) {
		//Activation
		fc2a[i] = actFn(fc2s[i]);
	}

	// Output
	for (int i = 0; i < 12; i++) {
		softout[i] = 0.0f;

		for (int j = 0; j < 128; j++) {
			softout[i] += fc2a[j] * w3[j][i];
		}
		softout[i] += biasw3[i];
	}

	// Softmax
	for (int i = 0; i < 12; i++) {
		float s = 0.f;
		// Total
		for (int j = 0; j < 12; j++) {
			s += exp(softout[j]);
		}
		softout2[i] = exp(softout[i]) / s;
	}

	// Print debugging

	//out += "kern1 weights\n";
	//for (int i = 0; i < 3; i++) {
	//	for (int j = 0; j < 3; j++) {
	//		for (int k = 0; k < 3; k++) {
	//			out += to_str(kern1[i][j][k][0]);
	//			out.push_back(' ');
	//		}
	//		out.push_back('\n');
	//	}
	//}

	//out += "kern1 weights\n";
	//for (int i = 0; i < 3; i++) {
	//	for (int j = 0; j < 3; j++) {
	//		for (int k = 0; k < 3; k++) {
	//			out += to_str(kern1[i][j][k][31]);
	//			out.push_back(' ');
	//		}
	//		out.push_back('\n');
	//	}
	//}

	//out += "\nconv1\n";
	//for (int i = 0; i < 32; i++) {
	//	for (int j = 0; j < 32; j++) {
	//		out += to_str(convL1[i][j][0]);
	//		out.push_back(' ');
	//	}
	//	out.push_back('\n');
	//}

	//out += "\nconv1\n";
	//for (int i = 0; i < 32; i++) {
	//	for (int j = 0; j < 32; j++) {
	//		out += to_str(convL1[i][j][31]);
	//		out.push_back(' ');
	//	}
	//	out.push_back('\n');
	//}

	//out += "\nmax1\n";
	//for (int i = 0; i < 16; i++) {
	//	for (int j = 0; j < 16; j++) {
	//		out += to_str(maxL1[i][j][0]);
	//		out.push_back(' ');
	//	}
	//	out.push_back('\n');
	//}

	//out += "\nmax1\n";
	//for (int i = 0; i < 16; i++) {
	//	for (int j = 0; j < 16; j++) {
	//		out += to_str(maxL1[i][j][31]);
	//		out.push_back(' ');
	//	}
	//	out.push_back('\n');
	//}

	//out += "\nmax1 index\n";
	//for (int i = 0; i < 16; i++) {
	//	for (int j = 0; j < 16; j++) {
	//		out += to_str(imaxL1[i][j][0]);
	//		out.push_back(' ');
	//	}
	//	out.push_back('\n');
	//}

	//out += "\nmax1 index\n";
	//for (int i = 0; i < 16; i++) {
	//	for (int j = 0; j < 16; j++) {
	//		out += to_str(imaxL1[i][j][31]);
	//		out.push_back(' ');
	//	}
	//	out.push_back('\n');
	//}

	//out += "\nconv2\n";
	//for (int i = 0; i < 14; i++) {
	//	for (int j = 0; j < 14; j++) {
	//		out += to_str(convL2[i][j][0]);
	//		out.push_back(' ');
	//	}
	//	out.push_back('\n');
	//}

	//out += "\nconv2\n";
	//for (int i = 0; i < 14; i++) {
	//	for (int j = 0; j < 14; j++) {
	//		out += to_str(convL2[i][j][63]);
	//		out.push_back(' ');
	//	}
	//	out.push_back('\n');
	//}

	//out += "\nmax2\n";
	//for (int i = 0; i < 7; i++) {
	//	for (int j = 0; j < 7; j++) {
	//		out += to_str(maxL2[i][j][0]);
	//		out.push_back(' ');
	//	}
	//	out.push_back('\n');
	//}

	//out += "\nmax2\n";
	//for (int i = 0; i < 7; i++) {
	//	for (int j = 0; j < 7; j++) {
	//		out += to_str(maxL2[i][j][63]);
	//		out.push_back(' ');
	//	}
	//	out.push_back('\n');
	//}

	//out += "\nmax2 index\n";
	//for (int i = 0; i < 7; i++) {
	//	for (int j = 0; j < 7; j++) {
	//		out += to_str(imaxL2[i][j][0]);
	//		out.push_back(' ');
	//	}
	//	out.push_back('\n');
	//}

	//out += "\nmax2 index\n";
	//for (int i = 0; i < 7; i++) {
	//	for (int j = 0; j < 7; j++) {
	//		out += to_str(imaxL2[i][j][63]);
	//		out.push_back(' ');
	//	}
	//	out.push_back('\n');
	//}

	//out += "\nl2 error\n";
	//for (int i = 0; i < 7; i++) {
	//	for (int j = 0; j < 7; j++) {
	//		out += to_str(emaxL2[i][j][0]);
	//		out.push_back(' ');
	//	}
	//	out.push_back('\n');
	//}

	//out += "\nl2 error\n";
	//for (int i = 0; i < 7; i++) {
	//	for (int j = 0; j < 7; j++) {
	//		out += to_str(emaxL2[i][j][63]);
	//		out.push_back(' ');
	//	}
	//	out.push_back('\n');
	//}

	//out += "\nl2 conv\n";
	//for (int i = 0; i < 14; i++) {
	//	for (int j = 0; j < 14; j++) {
	//		out += to_str(econvL2[i][j][0]);
	//		out.push_back(' ');
	//	}
	//	out.push_back('\n');
	//}

	//out += "\nl2 conv\n";
	//for (int i = 0; i < 14; i++) {
	//	for (int j = 0; j < 14; j++) {
	//		out += to_str(econvL2[i][j][63]);
	//		out.push_back(' ');
	//	}
	//	out.push_back('\n');
	//}

	//out += "\nl2 pad 2\n";
	//for (int i = 0; i < 18; i++) {
	//	for (int j = 0; j < 18; j++) {
	//		out += to_str(pconvL2[i][j][0]);
	//		out.push_back(' ');
	//	}
	//	out.push_back('\n');
	//}

	//out += "\nconv3\n";
	//for (int i = 0; i < 4; i++) {
	//	for (int j = 0; j < 4; j++) {
	//		out += to_str(convL3[i][j][0]);
	//		out.push_back(' ');
	//	}
	//	out.push_back('\n');
	//}

	//out += "\nconv3\n";
	//for (int i = 0; i < 4; i++) {
	//	for (int j = 0; j < 4; j++) {
	//		out += to_str(convL3[i][j][127]);
	//		out.push_back(' ');
	//	}
	//	out.push_back('\n');
	//}

	//out += "\nmax3\n";
	//out += to_str(maxL3[0][0][0]);
	//out.push_back(' ');
	//out += to_str(maxL3[0][1][0]);
	//out.push_back('\n');
	//out += to_str(maxL3[1][0][0]);
	//out.push_back(' ');
	//out += to_str(maxL3[1][1][0]);
	//out.push_back('\n');

	//out += "\nmax3\n";
	//out += to_str(maxL3[0][0][127]);
	//out.push_back(' ');
	//out += to_str(maxL3[0][1][127]);
	//out.push_back('\n');
	//out += to_str(maxL3[1][0][127]);
	//out.push_back(' ');
	//out += to_str(maxL3[1][1][127]);
	//out.push_back('\n');

	//out += "\nmax3 index\n";
	//out += to_str(imaxL3[0][0][0]);
	//out.push_back(' ');
	//out += to_str(imaxL3[0][1][0]);
	//out.push_back('\n');
	//out += to_str(imaxL3[1][0][0]);
	//out.push_back(' ');
	//out += to_str(imaxL3[1][1][0]);
	//out.push_back('\n');

	//out += "\nmax3 index\n";
	//out += to_str(imaxL3[0][0][127]);
	//out.push_back(' ');
	//out += to_str(imaxL3[0][1][127]);
	//out.push_back('\n');
	//out += to_str(imaxL3[1][0][127]);
	//out.push_back(' ');
	//out += to_str(imaxL3[1][1][127]);
	//out.push_back('\n');

	//// convL3 Errors
	//out += "\nconv3 error\n";
	//for (int i = 0; i < 4; i++) {
	//	for (int j = 0; j < 4; j++) {
	//		out += to_str(econvL3[i][j][0]);
	//		out.push_back(' ');
	//	}
	//	out.push_back('\n');
	//}
	//out += "\nconv3 error\n";
	//for (int i = 0; i < 4; i++) {
	//	for (int j = 0; j < 4; j++) {
	//		out += to_str(econvL3[i][j][127]);
	//		out.push_back(' ');
	//	}
	//	out.push_back('\n');
	//}

	//// convL3 Dialation
	//out += "\nconv3 dialate\n";
	//for (int i = 0; i < 7; i++) {
	//	for (int j = 0; j < 7; j++) {
	//		out += to_str(diconvL3[i][j][0]);
	//		out.push_back(' ');
	//	}
	//	out.push_back('\n');
	//}
	//out += "\nconv3 dialate\n";
	//for (int i = 0; i < 7; i++) {
	//	for (int j = 0; j < 7; j++) {
	//		out += to_str(diconvL3[i][j][127]);
	//		out.push_back(' ');
	//	}
	//	out.push_back('\n');
	//}

	//out += "\nfc1\n";
	//for (int i = 0; i < 128; i++) {
	//	out += to_str(fc1[i]);
	//	out.push_back(' ');
	//}

	//out += "\nfc2\n";
	//for (int i = 0; i < 128; i++) {
	//	out += to_str(fc2a[i]);
	//	out.push_back(' ');
	//}

	//out += "\nsoftmax\n";
	//for (int i = 0; i < 12; i++) {
	//	out += to_str(softout[i]);
	//	out.push_back(' ');
	//}
	//out.push_back('\n');

	out += to_str(ll);
	out += "\nsoftmax2\n";
	for (int i = 0; i < 12; i++) {
		out += to_str(softout2[i]);
		out.push_back(' ');
		if ((i + 1) % 4 == 0) out.push_back('\n');
	}
	out.push_back('\n');

	float expected[12] = { 0.0f };
	expected[img_class] = 1.0f;

	out += "\ncross entropy error: ";
	float ce = 0.0f;
	for (int i = 0; i < 12; i++) {
		ce += expected[i] * log(softout2[i]);
	}
	out += to_str(-ce);

	return max_element(softout2, softout2 + 12) - softout2;
}

void doBackProp(size_t ll, float image[65][65][3], int img_class, float anneal)
{
	float expected[12] = { 0.0f };
	expected[img_class] = 1.0f;

	// FC3 bias
	for (int i = 0; i < 12; i++) {
		// Cross Entropy derivative with softmax
		dbiasw3[i] = softout2[i] - expected[i];
	}

	// FC3 gradient
	for (int i = 0; i < 128; i++) {
		for (int j = 0; j < 12; j++) {
			// With respect to the output of previous layer
			dw3[i][j] = dbiasw3[j] * fc2a[i];
		}
	}

	// FC2 bias
	for (int i = 0; i < 128; i++) {
		dbiasw2[i] = 0.0f;
		for (int k = 0; k < 12; k++) {
			// With respect to w3
			dbiasw2[i] += dbiasw3[k] * w3[i][k];
		}
	}

	// FC2 gradient
	for (int i = 0; i < 128; i++) {
		for (int j = 0; j < 128; j++) {
			// With respect to the activation function of fc2 and the output of previous layer
			dw2[i][j] = dbiasw2[i] * dactFn(fc2s[i]) * fc1a[j];
		}
	}

	// FC1 bias
	for (int i = 0; i < 128; i++) {
		dbiasw1[i] = 0.0f;
		for (int k = 0; k < 128; k++) {
			// With respect to activation function of fc2 and w2
			dbiasw1[i] += dbiasw2[k] * dactFn(fc2s[k]) * w2[k][i];
		}
	}

	// FC1 gradient
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 128; k++) {
				for (int l = 0; l < 128; l++) {
					// With respect to activation function of fc1 and the output of previous layer
					dw1[i][j][k][l] = dbiasw1[l] * dactFn(fc1s[l]) * maxL3[i][j][k];
				}
			}
		}
	}

	// L3 error
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 128; k++) {
				emaxL3[i][j][k] = 0.0f;
				for (int l = 0; l < 128; l++) {
					// Figure out the delta outputs instead of delta weights
					emaxL3[i][j][k] += dbiasw1[l] * dactFn(fc1s[l]) * w1[i][j][k][l];
				}
			}
		}
	}

	// Kern3 bias 
	for (int i = 0; i < 128; i++) {
		dbias3[i] = 0.0;
		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 2; k++) {
				dbias3[i] += emaxL3[j][k][i];
			}
		}
	}

	// Restructure L3, 2x2 -> 4x4
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			int i0 = i / 2;
			int j0 = j / 2;
			for (int k = 0; k < 128; k++) {
				econvL3[i][j][k] = imaxL3[i0][j0][k] == i * 4 + j ?
					emaxL3[i0][j0][k] * dactFn(convL3s[j][i][k]) : 0.0f;
			}
		}
	}

	// Dialate L3 stride=2
	// https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-fb2f2efc4faa
	for (int i = 0; i < 7; i++) {
		for (int j = 0; j < 7; j++) {
			int i0 = i / 2;
			int j0 = j / 2;
			for (int k = 0; k < 128; k++) {
				//diconvL3[i][j][k] = ((i % 2 == 1) || (j % 2 == 1)) ?
				//	0.0f : econvL3[i0][j0][k] * dactFn(convL3s[j0][i0][k]);
				diconvL3[i][j][k] = ((i % 2 == 1) || (j % 2 == 1)) ?
					0.0f : econvL3[i0][j0][k] ;
			}
		}
	}

	// Kern3 gradient
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 64; k++) {
				for (int l = 0; l < 128; l++) {
					// Convolve 7x7 error over 9x9 input
					float s = 0.0f;
					for (int x = 0; x < 7; x++) {
						for (int y = 0; y < 7; y++) {
							int l2x = x + i - 1;
							int l2y = y + j - 1;
							// Padding
							bool b = l2x < 0 || l2y < 0 || l2x > 6 || l2y > 6;
							s += b ? 0.0f : diconvL3[x][y][l] * maxL2[l2x][l2y][k];
						}
					}
					dkern3[i][j][k][l] = s;
				}
			}
		}
	}

	// L2 error
	for (int i = 0; i < 7; i++) {
		for (int j = 0; j < 7; j++) {
			int i0 = i, i1 = i - 1, i2 = i + 1;
			int j0 = j, j1 = j - 1, j2 = j + 1;
			// Padding
			bool b0 = i1 < 0 || j1 < 0, b1 = i1 < 0, b2 = i1 < 0 || j2 > 6, b3 = j1 < 0;
			bool b4 = j2 > 6, b5 = i2 > 6 || j1 < 0, b6 = i2 > 6, b7 = i2 > 6 || j2 > 6;
			for (int k = 0; k < 64; k++) {
				float s = 0.0f;
				// Convolve flipped 3x3 filter over 7x7 error padded to 9x9
				for (int l = 0; l < 128; l++) {
					s += (b0 ? 0.0f : diconvL3[i1][j1][l] * kern3[2][2][k][l]);
					s += (b1 ? 0.0f : diconvL3[i1][j0][l] * kern3[2][1][k][l]);
					s += (b2 ? 0.0f : diconvL3[i1][j2][l] * kern3[2][0][k][l]);
					s += (b3 ? 0.0f : diconvL3[i0][j1][l] * kern3[1][2][k][l]);
					s += diconvL3[i0][j0][l] * kern3[1][1][k][l];
					s += (b4 ? 0.0f : diconvL3[i0][j2][l] * kern3[1][0][k][l]);
					s += (b5 ? 0.0f : diconvL3[i2][j1][l] * kern3[0][2][k][l]);
					s += (b6 ? 0.0f : diconvL3[i2][j0][l] * kern3[0][1][k][l]);
					s += (b7 ? 0.0f : diconvL3[i2][j2][l] * kern3[0][0][k][l]);
				}
				emaxL2[i][j][k] = s;
			}
		}
	}

	// Kern2 bias
	for (int i = 0; i < 64; i++) {
		dbias2[i] = 0.0f;
		for (int j = 0; j < 7; j++) {
			for (int k = 0; k < 7; k++) {
				dbias2[i] += emaxL2[j][k][i];
			}
		}
	}

	// Restructure L2, 7x7 -> 14x14
	for (int i = 0; i < 14; i++) {
		for (int j = 0; j < 14; j++) {
			int i0 = i / 2;
			int j0 = j / 2;
			for (int k = 0; k < 64; k++) {
				econvL2[i][j][k] = imaxL2[i0][j0][k] == i * 14 + j ?
					emaxL2[i0][j0][k] * dactFn(convL2s[j][i][k]) : 0.0f;
			}
		}
	}

	// Kern2 gradient
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 32; k++) {
				for (int l = 0; l < 64; l++) {
					// Convolve 14x14 error over 16x16 input
					float s = 0.0f;
					for (int x = 0; x < 14; x++) {
						for (int y = 0; y < 14; y++) {
							int l1x = x + i;
							int l1y = y + j;
							s += econvL2[x][y][l] * maxL1[l1x][l1y][k];
						}
					}
					dkern2[i][j][k][l] = s;
				}
			}
		}
	}

	// L2 error padded = 2
	for (int i = 0; i < 18; i++) {
		for (int j = 0; j < 18; j++) {
			for (int k = 0; k < 64; k++) {
				pconvL2[i][j][k] = i < 2 || j < 2 || i > 15 || j > 15 ?
					0.0f : econvL2[i - 2][j - 2][k];
				//if (pconvL2[i][j][k] * 1000000.0 != 0.f) {
				//	out += to_str(i);
				//	out += " ";
				//	out += to_str(j);
				//	out += " ";
				//	out += to_str(k);
				//	out += "\n";
				//}
			}
		}
	}

	// Kern1 error
	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++) {
			for (int k = 0; k < 32; k++) {
				float s = 0.0f;
				for (int l = 0; l < 64; l++) {
					s += pconvL2[i + 0][j + 0][l] * kern2[2][2][k][l];
					s += pconvL2[i + 0][j + 1][l] * kern2[2][1][k][l];
					s += pconvL2[i + 0][j + 2][l] * kern2[2][0][k][l];
					s += pconvL2[i + 1][j + 0][l] * kern2[1][2][k][l];
					s += pconvL2[i + 1][j + 1][l] * kern2[1][1][k][l];
					s += pconvL2[i + 1][j + 2][l] * kern2[1][0][k][l];
					s += pconvL2[i + 2][j + 0][l] * kern2[0][2][k][l];
					s += pconvL2[i + 2][j + 1][l] * kern2[0][1][k][l];
					s += pconvL2[i + 2][j + 2][l] * kern2[0][0][k][l];
				}
				emaxL1[i][j][k] = s;
			}
		}
	}

	// Kern1 bias
	for (int i = 0; i < 32; i++) {
		dbias1[i] = 0.0f;
		for (int j = 0; j < 16; j++) {
			for (int k = 0; k < 16; k++) {
				dbias1[i] += emaxL1[j][k][i];
			}
		}
	}

	// Restructure L1, 16x16 -> 32x32
	for (int i = 0; i < 32; i++) {
		for (int j = 0; j < 32; j++) {
			int i0 = i / 2;
			int j0 = j / 2;
			for (int k = 0; k < 32; k++) {
				econvL1[i][j][k] = imaxL1[i0][j0][k] == i * 32 + j ?
					emaxL1[i0][j0][k] * dactFn(convL1s[j][i][k]) : 0.0f;
			}
		}
	}

	// L1 dialation of stride=2
	for (int i = 0; i < 63; i++) {
		for (int j = 0; j < 63; j++) {
			int i0 = i / 2;
			int j0 = j / 2;
			for (int k = 0; k < 32; k++) {
				diconvL1[i][j][k] = ((i % 2 == 1) || (j % 2 == 1)) ?
					0.0f : econvL1[i0][j0][k];
				//if (econvL2[i][j][k] * 1000000.0 != 0.f) {
				//	out += to_str(i);
				//	out += " ";
				//	out += to_str(j);
				//	out += " ";
				//	out += to_str(k);
				//	out += "\n";
				//}
			}
		}
	}

	// Kern1 gradient
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 32; l++) {
					// Convolve 63x63 error over 65x65 input
					float s = 0.0f;
					for (int x = 0; x < 63; x++) {
						for (int y = 0; y < 63; y++) {
							int l1x = x + i;
							int l1y = y + j;
							s += image[l1x][l1y][k] * diconvL1[x][y][l];
							//s += image.at<Vec3b>(l1x, l1y)[k] / 255.0f * diconvL1[x][y][l];
						}
					}
					dkern1[i][j][k][l] = s;
				}
			}
		}
	}

	// Update step

	// FC3 weights
	for (int i = 0; i < 128; i++) {
		for (int j = 0; j < 12; j++) {
			w3[i][j] -= lr * anneal * dw3[i][j];
		}
	}

	// FC3 bias
	for (int i = 0; i < 12; i++) {
		biasw3[i] -= lrb * anneal * dbiasw3[i];
	}

	// FC2 weights
	for (int i = 0; i < 128; i++) {
		for (int j = 0; j < 128; j++) {
			w2[i][j] -= lr * 0.5f * anneal * dw2[i][j];
		}
	}

	// FC2 bias
	for (int i = 0; i < 128; i++) {
		biasw2[i] -= lrb * 0.5f * anneal * dbiasw2[i];
	}

	// FC1 weights
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 128; k++) {
				for (int l = 0; l < 128; l++) {
					w1[i][j][k][l] -= lr * 0.25f * anneal * dw1[i][j][k][l];
				}
			}
		}
	}

	// FC1 bias
	for (int i = 0; i < 128; i++) {
		biasw1[i] -= lrb * 0.25f * anneal * dbiasw1[i];
	}

	// Kern3 bias
	for (int i = 0; i < 128; i++) {
		bias3[i] -= lrb * 0.1f * anneal * dbias3[i];
	}

	// Kern3 weights
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 64; k++) {
				for (int l = 0; l < 128; l++) {
					kern3[i][j][k][l] -= lr * 0.1f * anneal * dkern3[i][j][k][l];
				}
			}
		}
	}

	// Kern2 bias
	for (int i = 0; i < 64; i++) {
		bias2[i] -= lrb * 0.05f * anneal * dbias2[i];
	}

	// Kern2 weights
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 32; k++) {
				for (int l = 0; l < 64; l++) {
					kern2[i][j][k][l] -= lr * 0.05f * anneal * dkern2[i][j][k][l];
				}
			}
		}
	}

	// Kern1 bias
	for (int i = 0; i < 32; i++) {
		bias1[i] -= lrb * 0.025f * anneal * dbias1[i];
	}

	// Kern1 weights
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 32; l++) {
					kern1[i][j][k][l] -= lr * 0.025f * anneal * dkern1[i][j][k][l];
				}
			}
		}
	}
}

int save(unordered_map<String, int> all_classes)
{
	String path = std::experimental::filesystem::current_path().string();
	ofstream outfile;
	outfile.open(path + "\\Weights.txt", ios::trunc);
	if (outfile.fail()) return -1;

	String o;

	o += "classes:\n";
	for (auto it : all_classes) {
		o += to_str(it.second) + " " +  it.first + " ";
	}
	o += "\n";

	o += "\nkern1:\n";
	for (int y = 0; y < 36; y++) {
		for (int x = 0; x < 24; x++) {
			int i = y % 3;
			int j = x % 3;
			int k = (y / 3) % 3;
			int l = (x / 3) + (y / 9) * 8;
			o += to_str(kern1[i][j][k][l]) + " ";
		}
		o += "\n";
	}

	o += "\nbias1:\n";
	for (int i = 0; i < 32; i++) {
		o += to_str(bias1[i]) + " ";
	}
	o += "\n";

	o += "\nkern2:\n";
	for (int y = 0; y < 192; y++) {
		for (int x = 0; x < 96; x++) {
			int i = y % 3;
			int j = x % 3;
			int k = x / 3;
			int l = y / 3;
			o += to_str(kern2[i][j][k][l]) + " ";
		}
		o += "\n";
	}

	o += "\nbias2:\n";
	for (int i = 0; i < 64; i++) {
		o += to_str(bias2[i]) + " ";
	}
	o += "\n";

	o += "\nkern3:\n";
	for (int y = 0; y < 384; y++) {
		for (int x = 0; x < 192; x++) {
			 int i = y % 3;
			 int j = x % 3;
			 int k = x / 3;
			 int l = y / 3;
			 o += to_str(kern3[i][j][k][l]) + " ";
		}
		o += "\n";
	}

	o += "\nbias3:\n";
	for (int i = 0; i < 128; i++) {
		o += to_str(bias3[i]) + " ";
	}
	o += "\n";

	o += "\nw1:\n";
	for (int y = 0; y < 256; y++) {
		for (int x = 0; x < 256; x++) {
			int i = y % 2;
			int j = x % 2;
			int k = y / 2;
			int l = x / 2;
			o += to_str(w1[i][j][k][l]) + " ";
		}
		o += "\n";
	}

	o += "\nbiasw1:\n";
	for (int i = 0; i < 128; i++) {
		o += to_str(biasw1[i]) + " ";
	}
	o += "\n";

	o += "\nw2:\n";
	for (int y = 0; y < 128; y++) {
		for (int x = 0; x < 128; x++) {
			o += to_str(w2[y][x]) + " ";
		}
		o += "\n";
	}

	o += "\nbiasw2:\n";
	for (int i = 0; i < 128; i++) {
		o += to_str(biasw2[i]) + " ";
	}
	o += "\n";

	o += "\nw3:\n";
	for (int y = 0; y < 64; y++) {
		for (int x = 0; x < 24; x++) {
			int i = y + (x / 12) * 64;
			int j = x % 12;
			o += to_str(w3[i][j]) + " ";
		}
		o += "\n";
	}

	o += "\nbiasw3:\n";
	for (int i = 0; i < 12; i++) {
		o += to_str(biasw3[i]) + " ";
	}
	o += "\n";

	outfile << o;
	outfile.close();

	return 0;
}

int main()
{
	random_device rd;
	default_random_engine gen{ rd() };
	uniform_real_distribution<float> dis0(0.0f, 64.0f);

	//// Setup input
	//for (int k = 0; k < 600; k++) {
	//	int r0 = floor(dis0(gen));
	//	int r1 = floor(dis0(gen));
	//	for (int i = 0; i < 65; i++) {
	//		for (int j = 0; j < 65; j++) {
	//			trainImg[k][i][j][0] = (i == r0 && j == r1) ? 1.0f : 0.0f;
	//			trainImg[k][i][j][1] = trainImg[k][i][j][0];
	//			trainImg[k][i][j][2] = trainImg[k][i][j][0];
	//		}
	//	}
	//	// XOR function
	//	trainClass[k] = ((r0 < 32) == (r1 < 32)) ? 0 : 1;
	//}

	//// Test input
	//for (int k = 0; k < 100; k++) {
	//	int r0 = floor(dis0(gen));
	//	int r1 = floor(dis0(gen));
	//	for (int i = 0; i < 65; i++) {
	//		for (int j = 0; j < 65; j++) {
	//			testImg[k][i][j][0] = (i == r0 && j == r1) ? 1.0f : 0.0f;
	//			testImg[k][i][j][1] = testImg[k][i][j][0];
	//			testImg[k][i][j][2] = testImg[k][i][j][0];
	//		}
	//	}
	//	// XOR function
	//	testClass[k] = ((r0 < 32) == (r1 < 32)) ? 0 : 1;
	//}

	/*
		Initialize weights to normal Gaussians with mean zero and
		standard deviation 1/sqrt(N_in) with N_in the cardinality of input
		connectivity into a next layer node
	*/
	normal_distribution<float> dis1(0.0f, 1.f / sqrt(27.f));
	// Random kern1 weights
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 32; l++) {
					kern1[i][j][k][l] = dis1(gen);
				}
			}
		}
	}

	//for (int i = 0; i < 3; i++) {
	//	for (int j = 0; j < 3; j++) {
	//		for (int k = 0; k < 3; k++) {
	//			for (int l = 0; l < 32; l++) {
	//				kern1[i][j][k][l] = i * j * k / (l + 1.0f);
	//			}
	//		}
	//	}
	//}

	// Bias for kern1
	for (int i = 0; i < 32; i++) {
		bias1[i] = 0.0f;
	}

	//for (int i = 0; i < 32; i++) {
	//	bias1[i] = i / 32.0f - 0.5f;
	//}

	normal_distribution<float> dis2(0.0f, 1.f / sqrt(288.f));
	// Random kern2 weights
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 32; k++) {
				for (int l = 0; l < 64; l++) {
					kern2[i][j][k][l] = dis2(gen);
				}
			}
		}
	}

	//for (int i = 0; i < 3; i++) {
	//	for (int j = 0; j < 3; j++) {
	//		for (int k = 0; k < 32; k++) {
	//			for (int l = 0; l < 64; l++) {
	//				kern2[i][j][k][l] = (i + j + k + l) / 1000.f;
	//			}
	//		}
	//	}
	//}

	// Bias for kern2
	for (int i = 0; i < 64; i++) {
		bias2[i] = 0.0f;
	}

	//for (int i = 0; i < 64; i++) {
	//	bias2[i] = 1.0f - (i / 64.0f) - 0.5f;
	//}

	normal_distribution<float> dis3(0.0f, 1.f / sqrt(576.f));
	// Random kern3 weights
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 64; k++) {
				for (int l = 0; l < 128; l++) {
					kern3[i][j][k][l] = dis3(gen);
				}
			}
		}
	}

	//for (int i = 0; i < 3; i++) {
	//	for (int j = 0; j < 3; j++) {
	//		for (int k = 0; k < 64; k++) {
	//			for (int l = 0; l < 128; l++) {
	//				kern3[i][j][k][l] = (i + j) / float(k + l + 1.0);
	//			}
	//		}
	//	}
	//}

	// Bias for kern3
	for (int i = 0; i < 128; i++) {
		bias3[i] = 0.0f;
	}

	//for (int i = 0; i < 128; i++) {
	//	bias3[i] = i / 128.0f - 0.5f;
	//}

	normal_distribution<float> dis4(0.0f, 1.f / sqrt(512.f));
	// FC1 random weights
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 128; k++) {
				for (int l = 0; l < 128; l++) {
					w1[i][j][k][l] = dis4(gen);
				}
			}
		}
	}

	//for (int i = 0; i < 2; i++) {
	//	for (int j = 0; j < 2; j++) {
	//		for (int k = 0; k < 128; k++) {
	//			for (int l = 0; l < 128; l++) {
	//				w1[i][j][k][l] = (i * (j + i)) * k / float(l * k + 1);
	//			}
	//		}
	//	}
	//}

	// Bias for FC1
	for (int i = 0; i < 128; i++) {
		biasw1[i] = 0.0f;
	}

	//for (int i = 0; i < 128; i++) {
	//	biasw1[i] = i % 8 / 8.0f;
	//}

	normal_distribution<float> dis5(0.0f, 1.f / sqrt(128.f));
	// FC2 random weights
	for (int i = 0; i < 128; i++) {
		for (int j = 0; j < 128; j++) {
			w2[i][j] = dis5(gen);
		}
	}

	//for (int i = 0; i < 128; i++) {
	//	for (int j = 0; j < 128; j++) {
	//		w2[i][j] = (i + j) / float(128*128);
	//	}
	//}

	// Bias for FC2
	for (int i = 0; i < 128; i++) {
		biasw2[i] = 0.0f;
	}

	//for (int i = 0; i < 128; i++) {
	//	biasw2[i] = 1.0f / (i + 1.0f);
	//}

	// FC3 random weights
	for (int i = 0; i < 128; i++) {
		for (int j = 0; j < 12; j++) {
			w3[i][j] = dis5(gen);
		}
	}

	//for (int i = 0; i < 128; i++) {
	//	for (int j = 0; j < 12; j++) {
	//		w3[i][j] = (i + j) / (100000000.0);
	//	}
	//}

	// Bias for FC3
	for (int i = 0; i < 12; i++) {
		biasw3[i] = 0.0f;
	}

	//for (int i = 0; i < 12; i++) {
	//	biasw3[i] = 1.0f - i / 12.0f;
	//}

	vector<String> fn;
	glob("D:\\Storage\\Datasets\\Train\\Fruits\\*.*", fn, true);

	vector<Mat> images;
	vector<int> img_class;
	size_t count = fn.size();

	unordered_map<String, int> all_classes;
	regex rgx("Fruits\\\\([A-Z]\\w+)");
	smatch matches;

	//pair<String, int> cur_class1("Apples", all_classes.size());
	//all_classes.insert(cur_class1);
	//pair<String, int> cur_class2("Bananas", all_classes.size());
	//all_classes.insert(cur_class2);
	//pair<String, int> cur_class3("Cherry", all_classes.size());
	//all_classes.insert(cur_class3);
	//pair<String, int> cur_class4("Grapes", all_classes.size());
	//all_classes.insert(cur_class4);
	//pair<String, int> cur_class5("Kiwi", all_classes.size());
	//all_classes.insert(cur_class5);
	//pair<String, int> cur_class6("Lemon", all_classes.size());
	//all_classes.insert(cur_class6);
	//pair<String, int> cur_class7("Onion", all_classes.size());
	//all_classes.insert(cur_class7);
	//pair<String, int> cur_class8("Orange", all_classes.size());
	//all_classes.insert(cur_class8);
	//pair<String, int> cur_class9("Tomato", all_classes.size());
	//all_classes.insert(cur_class9);
	//pair<String, int> cur_class10("Watermelon", all_classes.size());
	//all_classes.insert(cur_class10);

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

	//float lin_r = images[0].at<Vec3b>(0, 64)[2] / 255.0;
	//std::cout << lin_r << endl;

	//float lin_b = images[0].at<Vec3b>(0, 0)[0] / 255.0;
	//std::cout << lin_b << endl;

	//float lin_g = images[0].at<Vec3b>(64, 64)[1] / 255.0;
	//std::cout << lin_g << endl;

	// Training

	int const iters = 75;
	for (int e = 0; e < 10; e++) {

		// Shuffle
		auto seed = unsigned(time(0));
		srand(seed);
		random_shuffle(images.begin(), images.end());
		srand(seed);
		random_shuffle(img_class.begin(), img_class.end());

		for (size_t ll = 0; ll < iters; ll++) {

			float floatRBG[65][65][3];
			Mat img = images[ll];

			for (int i = 0; i < 65; i++) {
				for (int j = 0; j < 65; j++) {
					for (int k = 0; k < 3; k++) {
						floatRBG[i][j][k] = img.at<Vec3b>(i, j)[k] / 255.0f;
					}
				}
			}

			string out;
			// Time the neural net
			auto t1 = chrono::high_resolution_clock::now();
			doForwardProp(ll, floatRBG, img_class[ll], out);
			auto t2 = chrono::high_resolution_clock::now();
			doBackProp(ll, floatRBG, img_class[ll], 1);
			auto t3 = chrono::high_resolution_clock::now();

			auto duration = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
			out += "\nforward pass: ";
			out += to_str(duration);
			out += "ms";
			duration = chrono::duration_cast<chrono::milliseconds>(t3 - t2).count();
			out += "\nbackward pass: ";
			out += to_str(duration);
			out += "ms\n";
			std::cout << out << endl;
		}
	}

	vector<String> fn_test;
	glob("D:\\Storage\\Datasets\\Test\\Fruits\\*.*", fn_test, true);
	vector<Mat> images_test;
	vector<String> str_class_test;
	vector<int> img_class_test;
	size_t count_t = fn_test.size();

	for (size_t i = 0; i < count_t; i++) {
		images_test.push_back(imread(fn_test[i]));
		regex_search(fn_test[i], matches, rgx);
		img_class_test.push_back(all_classes.at(matches[1]));
		str_class_test.push_back(matches[1]);
	}

	int cc = 0;
	// Testing
	for (size_t ll = 0; ll < count_t; ll++) {

		float floatRBG[65][65][3];
		Mat img = images[ll];

		for (int i = 0; i < 65; i++) {
			for (int j = 0; j < 65; j++) {
				for (int k = 0; k < 3; k++) {
					floatRBG[i][j][k] = img.at<Vec3b>(i, j)[k] / 255.0f;
				}
			}
		}

		string out;
		int out_class = doForwardProp(ll, floatRBG, img_class_test[ll], out);
		cc = out_class == img_class_test[ll] ? cc + 1 : cc;
		out += "\nCorrect: " + to_str(cc) + " / " + to_str(count_t);
		std::cout << out << endl;
	}

	int s = save(all_classes);
	system("pause");
	return s;
}