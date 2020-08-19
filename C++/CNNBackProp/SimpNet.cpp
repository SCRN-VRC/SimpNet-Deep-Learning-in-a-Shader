
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <random>

using namespace std;
using namespace cv;
using namespace cv::ml;

#define DEBUG 1

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
	float lrb;
	float alpha;

	// forward prop

	// ouputs
	float*** L1s;		// 32 x 32 x 32
	float*** L1a;		// 32 x 32 x 32
	float*** L1Max;		// 16 x 16 x 32
	uint*** L1iMax;		// 16 x 16 x 32

	float*** L2s;		// 14 x 14 x 64
	float*** L2a;		// 14 x 14 x 64
	float*** L2Max;		// 7 x 7 x 64
	uint*** L2iMax;		// 7 x 7 x 64

	float*** L3s;		// 3 x 3 x 128
	float*** L3a;		// 3 x 3 x 128
	float* L3Max;		// 1 x 1 x 128
	uint* L3iMax;		// 1 x 1 x 128

	float* FC1s;		// 1 x 128
	float* FC1a;		// 1 x 128

	float* FC2s;		// 1 x 128
	float* FC2a;		// 1 x 128

	float* FC3s;		// 1 x 12
	float* FC3o;		// 1 x 12

	// kernels and weights
	float**** wL1;		// 3 x 3 x 3 x 32
	float* bL1;			// 1 x 32

	float**** wL2;		// 3 x 3 x 32 x 64
	float* bL2;			// 1 x 64

	float**** wL3;		// 3 x 3 x 64 x 128
	float* bL3;			// 1 x 128

	float** wFC1;		// 128 x 128
	float* bFC1;		// 1 x 128

	float** wFC2;		// 128 x 128
	float* bFC2;		// 1 x 128

	float** wFC3;		// 128 x 12
	float* bFC3;		// 1 x 12

	// backprop
	float** dwFC3;		// 128 x 12
	float* dbFC3;		// 1 x 12

	float** dwFC2;		// 128 x 128
	float* dbFC2;		// 1 x 128

	// Annoying malloc frees
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

public:

	// Annoying mallocs
	static void** createArray(int i, int j, size_t size)
	{
		void** r = (void**)malloc(i * sizeof(void*));
		for (int x = 0; x < i; x++) {
			r[x] = (void*)malloc(j * size);
		}
		return r;
	}

	static void*** createArray(int i, int j, int k, size_t size)
	{
		void*** r = (void***)malloc(i * sizeof(void*));
		for (int x = 0; x < i; x++) {
			r[x] = (void**)malloc(j * sizeof(void*));
			for (int y = 0; y < j; y++) {
				r[x][y] = (void*)malloc(k * size);
			}
		}
		return r;
	}

	static void**** createArray(int i, int j, int k, int l, size_t size)
	{
		void**** r = (void****)malloc(i * sizeof(void*));
		for (int x = 0; x < i; x++) {
			r[x] = (void***)malloc(j * sizeof(void*));
			for (int y = 0; y < j; y++) {
				r[x][y] = (void**)malloc(k * sizeof(void*));
				for (int z = 0; z < k; z++) {
					r[x][y][z] = (void*)malloc(l * size);
				}
			}
		}
		return r;
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
		lr = 0.2f;
		lrb = 0.01f;
		alpha = 0.1f;

		L1s = (float***)createArray(32, 32, 32, sizeof(float));
		L1a = (float***)createArray(32, 32, 32, sizeof(float));

		for (int i = 0; i < 32; i++) {
			for (int j = 0; j < 32; j++) {
				for (int k = 0; k < 32; k++) {
					L1s[i][j][k] = 0.0f;
					L1a[i][j][k] = 0.0f;
				}
			}
		}

		L1Max = (float***)createArray(16, 16, 32, sizeof(float));
		L1iMax = (uint***)createArray(16, 16, 32, sizeof(uint));

		for (int i = 0; i < 16; i++) {
			for (int j = 0; j < 16; j++) {
				for (int k = 0; k < 32; k++) {
					L1Max[i][j][k] = 0.0f;
					L1iMax[i][j][k] = 0;
				}
			}
		}

		L2s = (float***)createArray(14, 14, 64, sizeof(float));
		L2a = (float***)createArray(14, 14, 64, sizeof(float));


		for (int i = 0; i < 14; i++) {
			for (int j = 0; j < 14; j++) {
				for (int k = 0; k < 64; k++) {
					L2s[i][j][k] = 0.0f;
					L2a[i][j][k] = 0.0f;
				}
			}
		}

		L2Max = (float***)createArray(7, 7, 64, sizeof(float));
		L2iMax = (uint***)createArray(7, 7, 64, sizeof(uint));

		for (int i = 0; i < 7; i++) {
			for (int j = 0; j < 7; j++) {
				for (int k = 0; k < 64; k++) {
					L2Max[i][j][k] = 0.0f;
					L2iMax[i][j][k] = 0;
				}
			}
		}

		L3s = (float***)createArray(3, 3, 128, sizeof(float));
		L3a = (float***)createArray(3, 3, 128, sizeof(float));

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 128; k++) {
					L3s[i][j][k] = 0.0f;
					L3a[i][j][k] = 0.0f;
				}
			}
		}

		L3Max = (float*)malloc(128 * sizeof(float));
		L3iMax = (uint*)malloc(128 * sizeof(uint));

		for (int k = 0; k < 128; k++) {
			L3Max[k] = 0.0f;
			L3iMax[k] = 0;
		}

		FC1s = (float*)malloc(128 * sizeof(float));
		FC1a = (float*)malloc(128 * sizeof(float));
		FC2s = (float*)malloc(128 * sizeof(float));
		FC2a = (float*)malloc(128 * sizeof(float));

		for (int i = 0; i < 128; i++) {
			FC1s[i] = 0.0f;
			FC1a[i] = 0.0f;
			FC2s[i] = 0.0f;
			FC2a[i] = 0.0f;
		}

		FC3s = (float*)malloc(12 * sizeof(float));
		FC3o = (float*)malloc(12 * sizeof(float));

		for (int i = 0; i < 12; i++) {
			FC3s[i] = 0.0f;
			FC3o[i] = 0.0f;
		}

		wL1 = (float****)createArray(3, 3, 3, 32, sizeof(float));

		random_device rd;  //Will be used to obtain a seed for the random number engine
		mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
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

		wL2 = (float****)createArray(3, 3, 32, 64, sizeof(float));

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

		wL3 = (float****)createArray(3, 3, 64, 128, sizeof(float));

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

		wFC1 = (float**)createArray(128, 128, sizeof(float));
		wFC2 = (float**)createArray(128, 128, sizeof(float));
		dwFC2 = (float**)createArray(128, 128, sizeof(float));

		normal_distribution<float> dis4(0.0f, 1.0f / sqrt(128.0f));
		for (int i = 0; i < 128; i++) {
			for (int j = 0; j < 128; j++) {
				wFC1[i][j] = dis4(gen);
				wFC2[i][j] = dis4(gen);
				dwFC2[i][j] = 0.0f;
			}
		}

		wFC3 = (float**)createArray(128, 128, sizeof(float));
		dwFC3 = (float**)createArray(128, 128, sizeof(float));

		normal_distribution<float> dis5(0.0f, 1.0f / sqrt(128.0f));
		for (int i = 0; i < 128; i++) {
			for (int j = 0; j < 12; j++) {
				wFC3[i][j] = dis5(gen);
				dwFC3[i][j] = 0.0f;
			}
		}

		bL1 = (float*)malloc(32 * sizeof(float));
		for (int i = 0; i < 32; i++) {
			bL1[i] = 0.0f;
		}

		bL2 = (float*)malloc(64 * sizeof(float));
		for (int i = 0; i < 64; i++) {
			bL2[i] = 0.0f;
		}

		bL3 = (float*)malloc(128 * sizeof(float));
		for (int i = 0; i < 128; i++) {
			bL3[i] = 0.0f;
		}

		bFC1 = (float*)malloc(128 * sizeof(float));
		bFC2 = (float*)malloc(128 * sizeof(float));
		dbFC2 = (float*)malloc(128 * sizeof(float));
		for (int i = 0; i < 128; i++) {
			bFC1[i] = 0.0f;
			bFC2[i] = 0.0f;
			dbFC2[i] = 0.0f;
		}

		bFC3 = (float*)malloc(12 * sizeof(float));
		dbFC3 = (float*)malloc(12 * sizeof(float));
		for (int i = 0; i < 12; i++) {
			bFC3[i] = 0.0f;
			dbFC3[i] = 0.0f;
		}
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
		freeArray(128, 128, (void**)dwFC2);
		freeArray(128, 128, (void**)wFC3);
		freeArray(128, 128, (void**)dwFC3);
		free(bL1);
		free(bL2);
		free(bL3);
		free(bFC1);
		free(bFC2);
		free(dbFC2);
		free(bFC3);
		free(dbFC3);
	}

	int forwardProp(float*** image, int classNo, String &o)
	{
		// Convolutional layer 1, kernel=3x3, stride=2
#if (DEBUG)
		o += "\nConv Layer 1:\n";
#endif
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
#if (DEBUG)
					o += to_str(L1a[i][j][k]) + " ";
#endif
				}
#if (DEBUG)
				o += "\n";
#endif
			}
#if (DEBUG)
			o += "\n";
#endif
		}

		// Max pooling layer 1, size=2x2, stride=2

#if (DEBUG)
		o += "\nMax Layer 1:\n";
#endif
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

#if (DEBUG)
					o += to_str(L1Max[i][j][k]) + " ";
#endif
				}
#if (DEBUG)
				o += "\n";
#endif
			}
#if (DEBUG)
			o += "\n";
#endif
		}

		// Max pooling layer 1 index
#if (DEBUG)
		o += "\nMax Layer 1 Index:\n";
#endif
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

#if (DEBUG)
					o += to_str(L1iMax[i][j][k]) + " ";
#endif
				}
#if (DEBUG)
				o += "\n";
#endif
			}
#if (DEBUG)
			o += "\n";
#endif
		}

		// Convolutional layer 2, kernel=3x3, stride=1
#if (DEBUG)
		o += "\nConv Layer 2:\n";
#endif
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
#if (DEBUG)
					o += to_str(L2a[i][j][k]) + " ";
#endif
				}
#if (DEBUG)
				o += "\n";
#endif
			}
#if (DEBUG)
			o += "\n";
#endif
		}

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

		// Convolutional layer 3, kernel=3x3, stride=2
		for (int k = 0; k < 128; k++) {
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					L3s[i][j][k] = 0.0f;

					int i0 = i * 2, i1 = i + 1, i2 = i + 2;
					int j0 = j * 2, j1 = j + 1, j2 = j + 2;

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

					L3s[i][j][k] += bL2[k];
					L3a[i][j][k] = afn(L3s[i][j][k]);
				}
			}
		}

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

		for (int k = 0; k < 128; k++) {
			FC1s[k] = 0.0;

			for (int l = 0; l < 128; l++) {
				FC1s[k] += L3Max[l] * wFC1[l][k];
			}

			FC1s[k] += bFC1[k];
			FC1a[k] = afn(FC1s[k]);
		}

		for (int k = 0; k < 128; k++) {
			FC2s[k] = 0.0;

			for (int l = 0; l < 128; l++) {
				FC2s[k] += FC1a[l] * wFC2[l][k];
			}

			FC2s[k] += bFC2[k];
			FC2a[k] = afn(FC2s[k]);
		}

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

#if (DEBUG)
		o += "\nsoftmax\n";
		for (int i = 0; i < 12; i++) {
			o += to_str(FC3o[i]);
			o.push_back(' ');
		}
		o.push_back('\n');

		float expected[12] = { 0.0f };
		expected[classNo] = 1.0f;

		o += "\ncross entropy error: ";
		float ce = 0.0f;
		for (int i = 0; i < 12; i++) {
			ce += expected[i] * log(FC3o[i]);
		}
		o += to_str(-ce);
		o.push_back('\n');
#endif
		return max_element(FC3o, FC3o + 12) - FC3o;
	}

	void backProp(float*** image, int classNo, String &o)
	{
		float expected[12] = { 0.0f };
		expected[classNo] = 1.0f;

		//o += "\ndelta bias: ";
		for (int i = 0; i < 12; i++) {
			// Cross Entropy derivative with softmax
			dbFC3[i] = (FC3o[i] - expected[i]);
			//o += to_str(expected[i]) + " ";
		}

		//o += "\ndelta w: ";
		for (int i = 0; i < 128; i++) {
			for (int j = 0; j < 12; j++) {
				dwFC3[i][j] = dbFC3[j] * FC2a[i];
				//o += to_str(dwFC2[i][j]) + " ";
			}
			//o += "\n";
		}

		for (int i = 0; i < 128; i++) {
			dbFC2[i] = 0.f;
			for (int j = 0; j < 12; j++) {
				dbFC2[i] += dbFC3[j] * wFC3[i][j];
			}
		}
	}

	void update(String &o)
	{
		// FC2 bias
		for (int i = 0; i < 6; i++) {
			bFC3[i] -= lrb * dbFC3[i];
		}

		// FC2 weights
		for (int i = 0; i < 24; i++) {
			for (int j = 0; j < 6; j++) {
				wFC3[i][j] -= lr * dwFC3[i][j];
			}
		}

		for (int i = 0; i < 24; i++) {
			bFC2[i] -= lrb * dbFC2[i];
		}

		for (int i = 0; i < 128; i++) {
			for (int j = 0; j < 128; j++) {
				wFC2[i][j] -= lr * dwFC2[i][j];
			}
		}
	}
};

int main()
{
	CNN testCNN;
	String o;

	random_device rd;
	default_random_engine gen{ rd() };
	uniform_real_distribution<float> dis0(0.0f, 64.0f);

	const int imageSize[2] = { 65, 65 };

	const int c = 1;
	float**** input = (float****)CNN::createArray(c, imageSize[0], imageSize[1], 3, sizeof(float));
	int* inClass = (int*)malloc(c * sizeof(int));

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

	for (int i = 0; i < c; i++) {
		int classOut = testCNN.forwardProp(input[i], inClass[i], o);
		o += "\nclassify: " + to_str(classOut);
		o += "\nexpected: " + to_str(inClass[i]);
		//testCNN.backProp(input[i], inClass[i], o);
		//testCNN.update(o);
	}

	//const int t = 100;
	//float**** test = (float****)CNN::createArray(c, 65, 65, 3, sizeof(float));
	//int* testClass = (int*)malloc(t * sizeof(int));

	//int correct = 0;

	//// Setup input
	//for (int k = 0; k < t; k++) {
	//	int r0 = (int)floor(dis0(gen));
	//	int r1 = (int)floor(dis0(gen));
	//	for (int i = 0; i < 65; i++) {
	//		for (int j = 0; j < 65; j++) {
	//			test[k][i][j][0] = (i == r0 && j == r1) ? 1.0f : 0.0f;
	//			test[k][i][j][1] = test[k][i][j][0];
	//			test[k][i][j][2] = test[k][i][j][0];
	//		}
	//	}
	//	// XOR function
	//	testClass[k] = ((r0 < 4.5f) == (r1 < 4.5f)) ? 0 : 1;
	//}

	//for (int i = 0; i < t; i++) {
	//	o += "\n\ntesting: " + to_str(i) + "\n";
	//	int classOut = testCNN.forwardProp(test[i], testClass[i], o);
	//	correct = (classOut == inClass[i]) ? correct + 1 : correct;
	//	o += "\nclassify: " + to_str(classOut);
	//	o += "\nexpected: " + to_str(inClass[i]);
	//	o += "\ncorrect: " + to_str(correct) + "/" + to_str(t);
	//}

	std::cout << o << endl;

	system("pause");
	return 0;
}