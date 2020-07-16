#include <math.h>
#include <stdlib.h>
#include <random>
#include <iostream>
#include <string>
#include <chrono>

using namespace std;

inline int clamp(int x, int y, int z)
{
	return x < y ? y : x > z ? z : x;
}

inline float actFn(float x) {
	// Sigmoid
	// return 1.0f / (1.0f + exp(-x));
	// ELU
	return x >= 0.0f ? x : (0.15f * (exp(x) - 1.0f));
	// RELU
	// return fmaxf(0.0f, x);
}
inline float dactFn(float x) {
	// Sigmoid
	// return x * (1.0f - x);
	// ELU
	return x >= 0.0f ? 1.0f : exp(x) * 0.15f;
}

// Learning rate
float lr = 0.2f;
// Bias learning rate
float lrb = 0.2f;

float testImg[65][65][3] = { 0.0f };
float testOut[12] = {
	0.f, 0.f, 0.f, 0.f,
	0.f, 0.f, 0.f, 0.f,
	0.f, 0.f, 0.f, 1.f
};
// L1
float kern1[3][3][3][32]; // 3x3x3x32
float bias1[32];
float convL1[32][32][32] = { 0.0f }; // 32x32x32
float maxL1[16][16][32] = { 0.0f }; // 16x16x32
int imaxL1[16][16][32] = { 0 }; // 16x16x32
// L2
float kern2[3][3][32][64]; // 3x3x32x64
float bias2[64];
float convL2[14][14][64] = { 0.0f }; // 14x14x64
float maxL2[7][7][64] = { 0.0f }; // 7x7x64
int imaxL2[7][7][64] = { 0 }; // 7x7x64
// L3
float kern3[3][3][64][128]; // 3x3x64x128
float bias3[128];
float convL3[4][4][128] = { 0.0f }; // 4x4x128
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
float dw3[128][12] = { 0.0f };
float dbiasw3[12] = { 0.0f };
float dw2[128][128] = { 0.0f };
float dbiasw2[128] = { 0.0f };
float dw1[2][2][128][128] = { 0.0f };
float dbiasw1[128] = { 0.0f };
float emaxL3[2][2][128] = { 0.0f };
float econvL3[4][4][128] = { 0.0f };

int main()
{
	default_random_engine gen;
	normal_distribution<float> dis0(-0.5f, 0.5f);

	// Setup input
	for (int i = 0; i < 65; i++) {
		for (int j = 0; j < 65; j++) {
			testImg[i][j][0] = (i + 1) / 32.0f;
			testImg[i][j][1] = (65 - j + 1) / 32.0f;
			testImg[i][j][2] = ((i % 2) == ((j + 1) % 2)) ? 1.0f : 0.0f;
		}
	}

	/*
		Initialize weights to normal Gaussians with mean zero and 
		standard deviation 1/sqrt(N_in) with N_in the cardinality of input 
		connectivity into a next layer node
	*/
	normal_distribution<float> dis1(0.0f, 1.f/sqrt(27.f));
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

	// Bias for kern1
	for (int i = 0; i < 32; i++) {
		bias1[i] = dis0(gen);
	}

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

	// Bias for kern2
	for (int i = 0; i < 64; i++) {
		bias2[i] = dis0(gen);
	}

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

	// Bias for kern3
	for (int i = 0; i < 128; i++) {
		bias3[i] = dis0(gen);
	}

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

	// Bias for FC1
	for (int i = 0; i < 128; i++) {
		biasw1[i] = dis0(gen);
	}

	normal_distribution<float> dis5(0.0f, 1.f / sqrt(128.f));
	// FC2 random weights
	for (int i = 0; i < 128; i++) {
		for (int j = 0; j < 128; j++) {
			w2[i][j] = dis5(gen);
		}
	}

	// Bias for FC2
	for (int i = 0; i < 128; i++) {
		biasw2[i] = dis0(gen);
	}

	// FC3 random weights
	for (int i = 0; i < 128; i++) {
		for (int j = 0; j < 12; j++) {
			w3[i][j] = dis5(gen);
		}
	}

	// Bias for FC3
	for (int i = 0; i < 12; i++) {
		biasw3[i] = dis0(gen);
	}

	for (int ll = 0; ll < 1; ll++) {
		// Time the neural net
		auto t1 = chrono::high_resolution_clock::now();

		// Convolutional layer 1, kernel=3x3, stride=2
		for (int k = 0; k < 32; k++) {
			for (int i = 0; i < 32; i++) {
				for (int j = 0; j < 32; j++) {
					convL1[i][j][k] = 0.0f;

					int i0 = i * 2, i1 = i0 + 1, i2 = i0 + 2;
					int j0 = j * 2, j1 = j0 + 1, j2 = j0 + 2;

					// Sample image
					for (int l = 0; l < 3; l++) {
						convL1[i][j][k] +=
							testImg[i0][j0][l] * kern1[0][0][l][k] +
							testImg[i0][j1][l] * kern1[0][1][l][k] +
							testImg[i0][j2][l] * kern1[0][2][l][k] +
							testImg[i1][j0][l] * kern1[1][0][l][k] +
							testImg[i1][j1][l] * kern1[1][1][l][k] +
							testImg[i1][j2][l] * kern1[1][2][l][k] +
							testImg[i2][j0][l] * kern1[2][0][l][k] +
							testImg[i2][j1][l] * kern1[2][1][l][k] +
							testImg[i2][j2][l] * kern1[2][2][l][k];
					}

					// Bias
					convL1[i][j][k] += bias1[k];

					// Activation
					convL1[i][j][k] = actFn(convL1[i][j][k]);
				}
			}
		}

		// Max pooling layer 1, size=2x2, stride=2
		for (int k = 0; k < 32; k++) {
			for (int i = 0; i < 16; i++) {
				for (int j = 0; j < 16; j++) {
					int i0 = i * 2, i1 = i0 + 1;
					int j0 = j * 2, j1 = j0 + 1;

					float m = convL1[i0][j0][k];
					m = fmaxf(m, convL1[i0][j1][k]);
					m = fmaxf(m, convL1[i1][j0][k]);
					m = fmaxf(m, convL1[i1][j1][k]);
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

					float m = convL1[i0][j0][k];
					imaxL1[i][j][k] = i0 * 32 + j0;

					m = fmaxf(m, convL1[i0][j1][k]);
					imaxL1[i][j][k] = (m == convL1[i0][j1][k]) ?
						(i0 * 32 + j1) : imaxL1[i][j][k];

					m = fmaxf(m, convL1[i1][j0][k]);
					imaxL1[i][j][k] = (m == convL1[i1][j0][k]) ?
						(i1 * 32 + j0) : imaxL1[i][j][k];

					m = fmaxf(m, convL1[i1][j1][k]);
					imaxL1[i][j][k] = (m == convL1[i1][j1][k]) ?
						(i1 * 32 + j1) : imaxL1[i][j][k];
				}
			}
		}

		// Convolutional layer 2, kernel=3x3, stride=1
		for (int k = 0; k < 64; k++) {
			for (int i = 0; i < 14; i++) {
				for (int j = 0; j < 14; j++) {
					convL2[i][j][k] = 0.0f;

					int i0 = i, i1 = i + 1, i2 = i + 2;
					int j0 = j, j1 = j + 1, j2 = j + 2;

					for (int l = 0; l < 32; l++) {
						convL2[i][j][k] +=
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
					convL2[i][j][k] += bias2[k];

					// Activation
					convL2[i][j][k] = actFn(convL2[i][j][k]);
				}
			}
		}

		// Max pooling layer 2, size=2x2, stride=2
		for (int k = 0; k < 64; k++) {
			for (int i = 0; i < 7; i++) {
				for (int j = 0; j < 7; j++) {
					int i0 = i * 2, i1 = i0 + 1;
					int j0 = j * 2, j1 = j0 + 1;

					float m = convL2[i0][j0][k];
					m = fmaxf(m, convL2[i0][j1][k]);
					m = fmaxf(m, convL2[i1][j0][k]);
					m = fmaxf(m, convL2[i1][j1][k]);
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

					float m = convL2[i0][j0][k];
					imaxL2[i][j][k] = i0 * 14 + j0;
					m = fmaxf(m, convL2[i0][j1][k]);
					imaxL2[i][j][k] = (m == convL2[i0][j1][k]) ?
						(i0 * 14 + j1) : imaxL2[i][j][k];
					m = fmaxf(m, convL2[i1][j0][k]);
					imaxL2[i][j][k] = (m == convL2[i1][j0][k]) ?
						(i1 * 14 + j0) : imaxL2[i][j][k];
					m = fmaxf(m, convL2[i1][j1][k]);
					imaxL2[i][j][k] = (m == convL2[i1][j1][k]) ?
						(i1 * 14 + j1) : imaxL2[i][j][k];
				}
			}
		}

		// Convolutional layer 3, kernel=3x3, pad=1, stride=2
		for (int k = 0; k < 128; k++) {
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					convL3[i][j][k] = { 0.0f };

					int i0 = i * 2, i1 = i0 - 1, i2 = i0 + 1;
					int j0 = j * 2, j1 = j0 - 1, j2 = j0 + 1;

					// Padding
					bool bi1 = i1 < 0, bj1 = j1 < 0, bi2 = i2 > 6, bj2 = j2 > 6;
					bool b02 = bi1 || bj1, b03 = bi1 || bj2, b04 = bi2 || bj1, b05 = bi2 || bj2;

					for (int l = 0; l < 64; l++) {
						convL3[i][j][k] +=
							(b02 ? 0.0f : maxL2[i1][j1][l] * kern3[0][0][l][k]) +
							(bi1 ? 0.0f : maxL2[i1][j0][l] * kern3[0][1][l][k]) +
							(b03 ? 0.0f : maxL2[i1][j2][l] * kern3[0][2][l][k]) +
							(bj1 ? 0.0f : maxL2[i0][j1][l] * kern3[1][0][l][k]) +
							maxL2[i0][j0][l] * kern3[1][1][l * 4][k] +
							(bj2 ? 0.0f : maxL2[i0][j2][l] * kern3[1][2][l][k]) +
							(b04 ? 0.0f : maxL2[i2][j1][l] * kern3[2][0][l][k]) +
							(bi2 ? 0.0f : maxL2[i2][j0][l] * kern3[2][1][l][k]) +
							(b05 ? 0.0f : maxL2[i2][j2][l] * kern3[2][2][l][k]);
					}

					// Bias
					convL3[i][j][k] += bias3[k];

					// Activation
					convL3[i][j][k] = actFn(convL3[i][j][k]);
				}
			}
		}

		// Max pooling layer 3, size=2x2, stride=2
		for (int k = 0; k < 128; k++) {
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 2; j++) {
					int i0 = i * 2, i1 = i0 + 1;
					int j0 = j * 2, j1 = j0 + 1;

					float m = convL3[i0][j0][k];
					m = fmaxf(m, convL3[i0][j1][k]);
					m = fmaxf(m, convL3[i1][j0][k]);
					m = fmaxf(m, convL3[i1][j1][k]);
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

					float m = convL3[i0][j0][k];
					imaxL3[i][j][k] = i0 * 4 + j0;
					m = fmaxf(m, convL3[i0][j1][k]);
					imaxL3[i][j][k] = (m == convL3[i0][j1][k]) ?
						(i0 * 4 + j1) : imaxL3[i][j][k];
					m = fmaxf(m, convL3[i1][j0][k]);
					imaxL3[i][j][k] = (m == convL3[i1][j0][k]) ?
						(i1 * 4 + j0) : imaxL3[i][j][k];
					m = fmaxf(m, convL3[i1][j1][k]);
					imaxL3[i][j][k] = (m == convL3[i1][j1][k]) ?
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

		auto t2 = chrono::high_resolution_clock::now();

		// Backpropagation

		// FC3 bias
		for (int i = 0; i < 12; i++) {
			// Cross Entropy derivative with softmax
			dbiasw3[i] = (softout2[i] - testOut[i]);
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

		// maxL3 error
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				for (int k = 0; k < 128; k++) {
					emaxL3[i][j][k] = 0.0f;
					for (int l = 0; l < 128; l++) {
						// Figure out the delta outputs instead of delta weights
						emaxL3[i][j][k] += dbiasw1[l] * dactFn(fc1s[l]) * w1[i][j][l][k];
					}
				}
			}
		}

		// Restructure, 2x2 -> 4x4
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				for (int k = 0; k < 128; k++) {
					int i0 = i / 2;
					int j0 = j / 2;
					econvL3[i][j][k] = imaxL3[i0][j0][k] == i * 4 + j ?
						emaxL3[i0][j0][k] : 0.0f;
				}
			}
		}

		// Update step

		//// FC3 weights
		//for (int i = 0; i < 128; i++) {
		//	for (int j = 0; j < 12; j++) {
		//		w3[i][j] -= lr * dw3[i][j];
		//	}
		//}

		//// FC3 bias
		//for (int i = 0; i < 12; i++) {
		//	biasw3[i] -= lrb * dbiasw3[i];
		//}

		//// FC2 weights
		//for (int i = 0; i < 128; i++) {
		//	for (int j = 0; j < 128; j++) {
		//		w2[i][j] -= lr * dw2[i][j];
		//	}
		//}

		//// FC2 bias
		//for (int i = 0; i < 128; i++) {
		//	biasw2[i] -= lrb * dbiasw2[i];
		//}
		//
		//// FC1 weights
		//for (int i = 0; i < 2; i++) {
		//	for (int j = 0; j < 2; j++) {
		//		for (int k = 0; k < 128; k++) {
		//			for (int l = 0; l < 128; l++) {
		//				w1[i][j][k][l] -= lr * dw1[i][j][k][l];
		//			}
		//		}
		//	}
		//}

		//// FC1 bias
		//for (int i = 0; i < 128; i++) {
		//	biasw1[i] -= lrb * dbiasw1[i];
		//}

		auto t3 = chrono::high_resolution_clock::now();

		// Print debugging

		string out;
		//out += "kern1 weights\n";
		//for (int i = 0; i < 3; i++) {
		//	for (int j = 0; j < 3; j++) {
		//		for (int k = 0; k < 3; k++) {
		//			out += to_string(kern1[i][j][k][0]);
		//			out.push_back(' ');
		//		}
		//		out.push_back('\n');
		//	}
		//}

		//out += "kern1 weights\n";
		//for (int i = 0; i < 3; i++) {
		//	for (int j = 0; j < 3; j++) {
		//		for (int k = 0; k < 3; k++) {
		//			out += to_string(kern1[i][j][k][31]);
		//			out.push_back(' ');
		//		}
		//		out.push_back('\n');
		//	}
		//}

		//out += "\nconv1\n";
		//for (int i = 0; i < 32; i++) {
		//	for (int j = 0; j < 32; j++) {
		//		out += to_string(convL1[i][j][0]);
		//		out.push_back(' ');
		//	}
		//	out.push_back('\n');
		//}

		//out += "\nconv1\n";
		//for (int i = 0; i < 32; i++) {
		//	for (int j = 0; j < 32; j++) {
		//		out += to_string(convL1[i][j][31]);
		//		out.push_back(' ');
		//	}
		//	out.push_back('\n');
		//}

		//out += "\nmax1\n";
		//for (int i = 0; i < 16; i++) {
		//	for (int j = 0; j < 16; j++) {
		//		out += to_string(maxL1[i][j][0]);
		//		out.push_back(' ');
		//	}
		//	out.push_back('\n');
		//}

		//out += "\nmax1\n";
		//for (int i = 0; i < 16; i++) {
		//	for (int j = 0; j < 16; j++) {
		//		out += to_string(maxL1[i][j][31]);
		//		out.push_back(' ');
		//	}
		//	out.push_back('\n');
		//}

		//out += "\nmax1 index\n";
		//for (int i = 0; i < 16; i++) {
		//	for (int j = 0; j < 16; j++) {
		//		out += to_string(imaxL1[i][j][0]);
		//		out.push_back(' ');
		//	}
		//	out.push_back('\n');
		//}

		//out += "\nmax1 index\n";
		//for (int i = 0; i < 16; i++) {
		//	for (int j = 0; j < 16; j++) {
		//		out += to_string(imaxL1[i][j][31]);
		//		out.push_back(' ');
		//	}
		//	out.push_back('\n');
		//}

		//out += "\nconv2\n";
		//for (int i = 0; i < 14; i++) {
		//	for (int j = 0; j < 14; j++) {
		//		out += to_string(convL2[i][j][0]);
		//		out.push_back(' ');
		//	}
		//	out.push_back('\n');
		//}

		//out += "\nconv2\n";
		//for (int i = 0; i < 14; i++) {
		//	for (int j = 0; j < 14; j++) {
		//		out += to_string(convL2[i][j][63]);
		//		out.push_back(' ');
		//	}
		//	out.push_back('\n');
		//}

		//out += "\nmax2\n";
		//for (int i = 0; i < 7; i++) {
		//	for (int j = 0; j < 7; j++) {
		//		out += to_string(maxL2[i][j][0]);
		//		out.push_back(' ');
		//	}
		//	out.push_back('\n');
		//}

		//out += "\nmax2\n";
		//for (int i = 0; i < 7; i++) {
		//	for (int j = 0; j < 7; j++) {
		//		out += to_string(maxL2[i][j][63]);
		//		out.push_back(' ');
		//	}
		//	out.push_back('\n');
		//}

		//out += "\nmax2 index\n";
		//for (int i = 0; i < 7; i++) {
		//	for (int j = 0; j < 7; j++) {
		//		out += to_string(imaxL2[i][j][0]);
		//		out.push_back(' ');
		//	}
		//	out.push_back('\n');
		//}

		//out += "\nmax2 index\n";
		//for (int i = 0; i < 7; i++) {
		//	for (int j = 0; j < 7; j++) {
		//		out += to_string(imaxL2[i][j][63]);
		//		out.push_back(' ');
		//	}
		//	out.push_back('\n');
		//}

		//out += "\nconv3\n";
		//for (int i = 0; i < 4; i++) {
		//	for (int j = 0; j < 4; j++) {
		//		out += to_string(convL3[i][j][0]);
		//		out.push_back(' ');
		//	}
		//	out.push_back('\n');
		//}

		//out += "\nconv3\n";
		//for (int i = 0; i < 4; i++) {
		//	for (int j = 0; j < 4; j++) {
		//		out += to_string(convL3[i][j][127]);
		//		out.push_back(' ');
		//	}
		//	out.push_back('\n');
		//}

		//out += "\nmax3\n";
		//out += to_string(maxL3[0][0][0]);
		//out.push_back(' ');
		//out += to_string(maxL3[0][1][0]);
		//out.push_back('\n');
		//out += to_string(maxL3[1][0][0]);
		//out.push_back(' ');
		//out += to_string(maxL3[1][1][0]);
		//out.push_back('\n');

		//out += "\nmax3\n";
		//out += to_string(maxL3[0][0][127]);
		//out.push_back(' ');
		//out += to_string(maxL3[0][1][127]);
		//out.push_back('\n');
		//out += to_string(maxL3[1][0][127]);
		//out.push_back(' ');
		//out += to_string(maxL3[1][1][127]);
		//out.push_back('\n');

		out += "\nmax3 index\n";
		out += to_string(imaxL3[0][0][0]);
		out.push_back(' ');
		out += to_string(imaxL3[0][1][0]);
		out.push_back('\n');
		out += to_string(imaxL3[1][0][0]);
		out.push_back(' ');
		out += to_string(imaxL3[1][1][0]);
		out.push_back('\n');

		out += "\nmax3 index\n";
		out += to_string(imaxL3[0][0][127]);
		out.push_back(' ');
		out += to_string(imaxL3[0][1][127]);
		out.push_back('\n');
		out += to_string(imaxL3[1][0][127]);
		out.push_back(' ');
		out += to_string(imaxL3[1][1][127]);
		out.push_back('\n');

		// convL3 Errors
		out += "\nconv3 error\n";
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				out += to_string(econvL3[i][j][0]);
				out.push_back(' ');
			}
			out.push_back('\n');
		}
		out += "\nconv3 error\n";
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				out += to_string(econvL3[i][j][127]);
				out.push_back(' ');
			}
			out.push_back('\n');
		}

		//out += "\nfc1\n";
		//for (int i = 0; i < 128; i++) {
		//	out += to_string(fc1[i]);
		//	out.push_back(' ');
		//}

		//out += "\nfc2\n";
		//for (int i = 0; i < 128; i++) {
		//	out += to_string(fc2a[i]);
		//	out.push_back(' ');
		//}

		out += "\nsoftmax\n";
		for (int i = 0; i < 12; i++) {
			out += to_string(softout[i]);
			out.push_back(' ');
		}
		out.push_back('\n');

		out += "\nsoftmax2\n";
		for (int i = 0; i < 12; i++) {
			out += to_string(softout2[i]);
			out.push_back(' ');
		}
		out.push_back('\n');

		out += "\ncross entropy error: ";
		float ce = 0.0f;
		for (int i = 0; i < 12; i++) {
			ce += testOut[i] * log(softout2[i]);
		}
		out += to_string(-ce);
		auto duration = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
		out += "\nforward pass: ";
		out += to_string(duration);
		out += "ms";
		duration = chrono::duration_cast<chrono::milliseconds>(t3 - t2).count();
		out += "\nbackward pass: ";
		out += to_string(duration);
		out += "ms\n";
		cout << out << endl;
		//system("pause");
	}
	system("pause");
	return 0;
}