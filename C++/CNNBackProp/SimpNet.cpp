#include <math.h>
#include <stdlib.h>
#include <random>
#include <iostream>
#include <string>
#include <chrono>

using namespace std;

typedef struct {
	float x;
	float y;
	float z;
	float w;
} float4;

typedef struct {
	int x;
	int y;
	int z;
	int w;
} int4;

inline int clamp(int x, int y, int z)
{
	return x < y ? y : x > z ? z : x;
}

inline float fnRELU(float x) {
	return fmaxf(0.0f, x);
}

inline float fnSig(float x) {
	return exp(x);
}

inline float fnELU(float x, float alpha) {
	return (float) (x >= 0.0f ? x : (alpha * (exp(x) - 1.0f)));
}

float4 testImg[65][65] = { 0.0f };
// L1
float4 kern1[3][3][3][8]; // 3x3x3x32
float4 bias1[8];
float4 convL1[32][32][8] = { 0.0f }; // 32x32x32
float4 maxL1[16][16][8] = { 0.0f }; // 16x16x32
int4 imaxL1[16][16][8] = { 0 }; // 16x16x32
// L2
float4 kern2[3][3][32][16]; // 3x3x32x64
float4 bias2[16];
float4 convL2[14][14][16] = { 0.0f }; // 14x14x64
float4 maxL2[7][7][16] = { 0.0f }; // 7x7x64
int4 imaxL2[7][7][16] = { 0 }; // 7x7x64
// L3
float4 kern3[3][3][64][32]; // 3x3x64x128
float4 bias3[32];
float4 convL3[4][4][32] = { 0.0f }; // 4x4x128
float4 maxL3[128] = { 0.0f }; // 2x2x128
int4 imaxL3[128] = { 0 }; // 2x2x128
// L4
float4 w1[512][32]; // 512x128
float4 biasw1[32];
float4 fc1[32] = { 0.0f }; // 1x1x128
// L5
float4 w2[128][2]; // 128x8
float4 biasw2[2];
float4 out[2] = { 0.0f }; // 1x1x8

int main()
{
	random_device rd;  //Will be used to obtain a seed for the random number engine
	mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	uniform_real_distribution<float> dis(-1.0f, 1.0f);

	// Setup input
	for (int i = 0; i < 65; i++) {
		for (int j = 0; j < 65; j++) {
			testImg[i][j].x = (i + 1) / 32.0f;
			testImg[i][j].y = (65 - j + 1) / 32.0f;
			testImg[i][j].z = ((i % 2) == ((j + 1) % 2)) ? 1.0f : 0.0f;
			testImg[i][j].z = 0.0f;
		}
	}

	// Random kern1 weights
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 8; l++) {
					kern1[i][j][k][l].x = dis(gen);
					kern1[i][j][k][l].y = dis(gen);
					kern1[i][j][k][l].z = dis(gen);
					kern1[i][j][k][l].w = dis(gen);
				}
			}
		}
	}

	// Bias for kern1
	for (int i = 0; i < 8; i++) {
		bias1[i].x = dis(gen);
		bias1[i].y = dis(gen);
		bias1[i].z = dis(gen);
		bias1[i].w = dis(gen);
	}

	// Random kern2 weights
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 32; k++) {
				for (int l = 0; l < 16; l++) {
					kern2[i][j][k][l].x = dis(gen);
					kern2[i][j][k][l].y = dis(gen);
					kern2[i][j][k][l].z = dis(gen);
					kern2[i][j][k][l].w = dis(gen);
				}
			}
		}
	}

	// Bias for kern2
	for (int i = 0; i < 16; i++) {
		bias2[i].x = dis(gen);
		bias2[i].y = dis(gen);
		bias2[i].z = dis(gen);
		bias2[i].w = dis(gen);
	}

	// Random kern3 weights
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 64; k++) {
				for (int l = 0; l < 32; l++) {
					kern3[i][j][k][l].x = dis(gen);
					kern3[i][j][k][l].y = dis(gen);
					kern3[i][j][k][l].z = dis(gen);
					kern3[i][j][k][l].w = dis(gen);
				}
			}
		}
	}

	// Bias for kern3
	for (int i = 0; i < 32; i++) {
		bias3[i].x = dis(gen);
		bias3[i].y = dis(gen);
		bias3[i].z = dis(gen);
		bias3[i].w = dis(gen);
	}

	// FC1 random weights
	for (int i = 0; i < 512; i++) {
		for (int j = 0; j < 16; j++) {
			w1[i][j].x = dis(gen);
			w1[i][j].y = dis(gen);
			w1[i][j].z = dis(gen);
			w1[i][j].w = dis(gen);
		}
	}

	// Bias for FC1
	for (int i = 0; i < 16; i++) {
		biasw1[i].x = dis(gen);
		biasw1[i].y = dis(gen);
		biasw1[i].z = dis(gen);
		biasw1[i].w = dis(gen);
	}

	// FC2 random weights
	for (int i = 0; i < 64; i++) {
		for (int j = 0; j < 2; j++) {
			w2[i][j].x = dis(gen);
			w2[i][j].y = dis(gen);
			w2[i][j].z = dis(gen);
			w2[i][j].w = dis(gen);
		}
	}

	// Bias for FC2
	for (int i = 0; i < 2; i++) {
		biasw2[i].x = dis(gen);
		biasw2[i].y = dis(gen);
		biasw2[i].z = dis(gen);
		biasw2[i].w = dis(gen);
	}

	// Time the neural net
	auto t1 = chrono::high_resolution_clock::now();

	// Convolutional layer 1, kernel=3x3, stride=2
	for (int k = 0; k < 8; k++) {
		for (int i = 0; i < 32; i++) {
			for (int j = 0; j < 32; j++) {
				int i0 = i * 2, i1 = i0 + 1, i2 = i0 + 2;
				int j0 = j * 2, j1 = j0 + 1, j2 = j0 + 2;

				// Sample
				// X
				convL1[i][j][k].x +=
					testImg[i0][j0].x * kern1[0][0][0][k].x +
					testImg[i0][j1].x * kern1[0][1][0][k].x +
					testImg[i0][j2].x * kern1[0][2][0][k].x +
					testImg[i1][j0].x * kern1[1][0][0][k].x +
					testImg[i1][j1].x * kern1[1][1][0][k].x +
					testImg[i1][j2].x * kern1[1][2][0][k].x +
					testImg[i2][j0].x * kern1[2][0][0][k].x +
					testImg[i2][j1].x * kern1[2][1][0][k].x +
					testImg[i2][j2].x * kern1[2][2][0][k].x;
				convL1[i][j][k].x +=
					testImg[i0][j0].y * kern1[0][0][1][k].x +
					testImg[i0][j1].y * kern1[0][1][1][k].x +
					testImg[i0][j2].y * kern1[0][2][1][k].x +
					testImg[i1][j0].y * kern1[1][0][1][k].x +
					testImg[i1][j1].y * kern1[1][1][1][k].x +
					testImg[i1][j2].y * kern1[1][2][1][k].x +
					testImg[i2][j0].y * kern1[2][0][1][k].x +
					testImg[i2][j1].y * kern1[2][1][1][k].x +
					testImg[i2][j2].y * kern1[2][2][1][k].x;
				convL1[i][j][k].x +=
					testImg[i0][j0].z * kern1[0][0][2][k].x +
					testImg[i0][j1].z * kern1[0][1][2][k].x +
					testImg[i0][j2].z * kern1[0][2][2][k].x +
					testImg[i1][j0].z * kern1[1][0][2][k].x +
					testImg[i1][j1].z * kern1[1][1][2][k].x +
					testImg[i1][j2].z * kern1[1][2][2][k].x +
					testImg[i2][j0].z * kern1[2][0][2][k].x +
					testImg[i2][j1].z * kern1[2][1][2][k].x +
					testImg[i2][j2].z * kern1[2][2][2][k].x;
				// Y
				convL1[i][j][k].y +=
					testImg[i0][j0].x * kern1[0][0][0][k].y +
					testImg[i0][j1].x * kern1[0][1][0][k].y +
					testImg[i0][j2].x * kern1[0][2][0][k].y +
					testImg[i1][j0].x * kern1[1][0][0][k].y +
					testImg[i1][j1].x * kern1[1][1][0][k].y +
					testImg[i1][j2].x * kern1[1][2][0][k].y +
					testImg[i2][j0].x * kern1[2][0][0][k].y +
					testImg[i2][j1].x * kern1[2][1][0][k].y +
					testImg[i2][j2].x * kern1[2][2][0][k].y;
				convL1[i][j][k].y +=
					testImg[i0][j0].y * kern1[0][0][1][k].y +
					testImg[i0][j1].y * kern1[0][1][1][k].y +
					testImg[i0][j2].y * kern1[0][2][1][k].y +
					testImg[i1][j0].y * kern1[1][0][1][k].y +
					testImg[i1][j1].y * kern1[1][1][1][k].y +
					testImg[i1][j2].y * kern1[1][2][1][k].y +
					testImg[i2][j0].y * kern1[2][0][1][k].y +
					testImg[i2][j1].y * kern1[2][1][1][k].y +
					testImg[i2][j2].y * kern1[2][2][1][k].y;
				convL1[i][j][k].y +=
					testImg[i0][j0].z * kern1[0][0][2][k].y +
					testImg[i0][j1].z * kern1[0][1][2][k].y +
					testImg[i0][j2].z * kern1[0][2][2][k].y +
					testImg[i1][j0].z * kern1[1][0][2][k].y +
					testImg[i1][j1].z * kern1[1][1][2][k].y +
					testImg[i1][j2].z * kern1[1][2][2][k].y +
					testImg[i2][j0].z * kern1[2][0][2][k].y +
					testImg[i2][j1].z * kern1[2][1][2][k].y +
					testImg[i2][j2].z * kern1[2][2][2][k].y;
				// Z
				convL1[i][j][k].z +=
					testImg[i0][j0].x * kern1[0][0][0][k].z +
					testImg[i0][j1].x * kern1[0][1][0][k].z +
					testImg[i0][j2].x * kern1[0][2][0][k].z +
					testImg[i1][j0].x * kern1[1][0][0][k].z +
					testImg[i1][j1].x * kern1[1][1][0][k].z +
					testImg[i1][j2].x * kern1[1][2][0][k].z +
					testImg[i2][j0].x * kern1[2][0][0][k].z +
					testImg[i2][j1].x * kern1[2][1][0][k].z +
					testImg[i2][j2].x * kern1[2][2][0][k].z;
				convL1[i][j][k].z +=
					testImg[i0][j0].y * kern1[0][0][1][k].z +
					testImg[i0][j1].y * kern1[0][1][1][k].z +
					testImg[i0][j2].y * kern1[0][2][1][k].z +
					testImg[i1][j0].y * kern1[1][0][1][k].z +
					testImg[i1][j1].y * kern1[1][1][1][k].z +
					testImg[i1][j2].y * kern1[1][2][1][k].z +
					testImg[i2][j0].y * kern1[2][0][1][k].z +
					testImg[i2][j1].y * kern1[2][1][1][k].z +
					testImg[i2][j2].y * kern1[2][2][1][k].z;
				convL1[i][j][k].z +=
					testImg[i0][j0].z * kern1[0][0][2][k].z +
					testImg[i0][j1].z * kern1[0][1][2][k].z +
					testImg[i0][j2].z * kern1[0][2][2][k].z +
					testImg[i1][j0].z * kern1[1][0][2][k].z +
					testImg[i1][j1].z * kern1[1][1][2][k].z +
					testImg[i1][j2].z * kern1[1][2][2][k].z +
					testImg[i2][j0].z * kern1[2][0][2][k].z +
					testImg[i2][j1].z * kern1[2][1][2][k].z +
					testImg[i2][j2].z * kern1[2][2][2][k].z;
				// W
				convL1[i][j][k].w +=
					testImg[i0][j0].x * kern1[0][0][0][k].w +
					testImg[i0][j1].x * kern1[0][1][0][k].w +
					testImg[i0][j2].x * kern1[0][2][0][k].w +
					testImg[i1][j0].x * kern1[1][0][0][k].w +
					testImg[i1][j1].x * kern1[1][1][0][k].w +
					testImg[i1][j2].x * kern1[1][2][0][k].w +
					testImg[i2][j0].x * kern1[2][0][0][k].w +
					testImg[i2][j1].x * kern1[2][1][0][k].w +
					testImg[i2][j2].x * kern1[2][2][0][k].w;
				convL1[i][j][k].w +=
					testImg[i0][j0].y * kern1[0][0][1][k].w +
					testImg[i0][j1].y * kern1[0][1][1][k].w +
					testImg[i0][j2].y * kern1[0][2][1][k].w +
					testImg[i1][j0].y * kern1[1][0][1][k].w +
					testImg[i1][j1].y * kern1[1][1][1][k].w +
					testImg[i1][j2].y * kern1[1][2][1][k].w +
					testImg[i2][j0].y * kern1[2][0][1][k].w +
					testImg[i2][j1].y * kern1[2][1][1][k].w +
					testImg[i2][j2].y * kern1[2][2][1][k].w;
				convL1[i][j][k].w +=
					testImg[i0][j0].z * kern1[0][0][2][k].w +
					testImg[i0][j1].z * kern1[0][1][2][k].w +
					testImg[i0][j2].z * kern1[0][2][2][k].w +
					testImg[i1][j0].z * kern1[1][0][2][k].w +
					testImg[i1][j1].z * kern1[1][1][2][k].w +
					testImg[i1][j2].z * kern1[1][2][2][k].w +
					testImg[i2][j0].z * kern1[2][0][2][k].w +
					testImg[i2][j1].z * kern1[2][1][2][k].w +
					testImg[i2][j2].z * kern1[2][2][2][k].w;

				// Bias
				convL1[i][j][k].x += bias1[k].x;
				convL1[i][j][k].y += bias1[k].y;
				convL1[i][j][k].z += bias1[k].z;
				convL1[i][j][k].w += bias1[k].w;

				// Activation
				convL1[i][j][k].x = fnELU(convL1[i][j][k].x, 0.15f);
				convL1[i][j][k].y = fnELU(convL1[i][j][k].y, 0.15f);
				convL1[i][j][k].z = fnELU(convL1[i][j][k].z, 0.15f);
				convL1[i][j][k].w = fnELU(convL1[i][j][k].w, 0.15f);
			}
		}
	}

	// Max pooling layer 1, size=2x2, stride=2
	for (int k = 0; k < 8; k++) {
		for (int i = 0; i < 16; i++) {
			for (int j = 0; j < 16; j++) {
				int i0 = i * 2, i1 = i0 + 1;
				int j0 = j * 2, j1 = j0 + 1;

				float m = convL1[i0][j0][k].x;
				m = fmaxf(m, convL1[i0][j1][k].x);
				m = fmaxf(m, convL1[i1][j0][k].x);
				m = fmaxf(m, convL1[i1][j1][k].x);
				maxL1[i][j][k].x = m;

				m = convL1[i0][j0][k].y;
				m = fmaxf(m, convL1[i0][j1][k].y);
				m = fmaxf(m, convL1[i1][j0][k].y);
				m = fmaxf(m, convL1[i1][j1][k].y);
				maxL1[i][j][k].y = m;

				m = convL1[i0][j0][k].z;
				m = fmaxf(m, convL1[i0][j1][k].z);
				m = fmaxf(m, convL1[i1][j0][k].z);
				m = fmaxf(m, convL1[i1][j1][k].z);
				maxL1[i][j][k].z = m;

				m = convL1[i0][j0][k].w;
				m = fmaxf(m, convL1[i0][j1][k].w);
				m = fmaxf(m, convL1[i1][j0][k].w);
				m = fmaxf(m, convL1[i1][j1][k].w);
				maxL1[i][j][k].w = m;
			}
		}
	}

	// Max pooling layer 1 index
	for (int k = 0; k < 8; k++) {
		for (int i = 0; i < 16; i++) {
			for (int j = 0; j < 16; j++) {
				int i0 = i * 2, i1 = i0 + 1;
				int j0 = j * 2, j1 = j0 + 1;

				// X
				float m = convL1[i0][j0][k].x;
				imaxL1[i][j][k].x = i0 * 32 + j0;

				m = fmaxf(m, convL1[i0][j1][k].x);
				imaxL1[i][j][k].x = (m == convL1[i0][j1][k].x) ?
					(i0 * 32 + j1) : imaxL1[i][j][k].x;

				m = fmaxf(m, convL1[i1][j0][k].x);
				imaxL1[i][j][k].x = (m == convL1[i1][j0][k].x) ?
					(i1 * 32 + j0) : imaxL1[i][j][k].x;

				m = fmaxf(m, convL1[i1][j1][k].x);
				imaxL1[i][j][k].x = (m == convL1[i1][j1][k].x) ?
					(i1 * 32 + j1) : imaxL1[i][j][k].x;

				// Y
				m = convL1[i0][j0][k].y;
				imaxL1[i][j][k].y = i0 * 32 + j0;

				m = fmaxf(m, convL1[i0][j1][k].y);
				imaxL1[i][j][k].y = (m == convL1[i0][j1][k].y) ?
					(i0 * 32 + j1) : imaxL1[i][j][k].y;

				m = fmaxf(m, convL1[i1][j0][k].y);
				imaxL1[i][j][k].y = (m == convL1[i1][j0][k].y) ?
					(i1 * 32 + j0) : imaxL1[i][j][k].y;

				m = fmaxf(m, convL1[i1][j1][k].y);
				imaxL1[i][j][k].y = m == (convL1[i1][j1][k].y) ?
					(i1 * 32 + j1) : imaxL1[i][j][k].y;

				// Z
				m = convL1[i0][j0][k].z;
				imaxL1[i][j][k].z = i0 * 32 + j0;

				m = fmaxf(m, convL1[i0][j1][k].z);
				imaxL1[i][j][k].z = (m == convL1[i0][j1][k].z) ?
					(i0 * 32 + j1) : imaxL1[i][j][k].z;

				m = fmaxf(m, convL1[i1][j0][k].z);
				imaxL1[i][j][k].z = (m == convL1[i1][j0][k].z) ?
					(i1 * 32 + j0) : imaxL1[i][j][k].z;

				m = fmaxf(m, convL1[i1][j1][k].z);
				imaxL1[i][j][k].z = (m == convL1[i1][j1][k].z) ?
					(i1 * 32 + j1) : imaxL1[i][j][k].z;

				// W
				m = convL1[i0][j0][k].w;
				imaxL1[i][j][k].w = i0 * 32 + j0;

				m = fmaxf(m, convL1[i0][j1][k].w);
				imaxL1[i][j][k].w = (m == convL1[i0][j1][k].w) ?
					(i0 * 32 + j1) : imaxL1[i][j][k].w;

				m = fmaxf(m, convL1[i1][j0][k].w);
				imaxL1[i][j][k].w = (m == convL1[i1][j0][k].w) ?
					(i1 * 32 + j0) : imaxL1[i][j][k].w;

				m = fmaxf(m, convL1[i1][j1][k].w);
				imaxL1[i][j][k].w = (m == convL1[i1][j1][k].w) ?
					(i1 * 32 + j1) : imaxL1[i][j][k].w;
			}
		}
	}

	// Convolutional layer 2, kernel=3x3, stride=1
	for (int k = 0; k < 16; k++) {
		for (int i = 0; i < 14; i++) {
			for (int j = 0; j < 14; j++) {
				int i0 = i, i1 = i + 1, i2 = i + 2;
				int j0 = j, j1 = j + 1, j2 = j + 2;

				for (int l = 0; l < 8; l++) {
					// X
					convL2[i][j][k].x +=
						maxL1[i0][j0][l].x * kern2[0][0][l * 4][k].x +
						maxL1[i0][j1][l].x * kern2[0][1][l * 4][k].x +
						maxL1[i0][j2][l].x * kern2[0][2][l * 4][k].x +
						maxL1[i1][j0][l].x * kern2[1][0][l * 4][k].x +
						maxL1[i1][j1][l].x * kern2[1][1][l * 4][k].x +
						maxL1[i1][j2][l].x * kern2[1][2][l * 4][k].x +
						maxL1[i2][j0][l].x * kern2[2][0][l * 4][k].x +
						maxL1[i2][j1][l].x * kern2[2][1][l * 4][k].x +
						maxL1[i2][j2][l].x * kern2[2][2][l * 4][k].x;
					convL2[i][j][k].x +=
						maxL1[i0][j0][l].y * kern2[0][0][l * 4 + 1][k].x +
						maxL1[i0][j1][l].y * kern2[0][1][l * 4 + 1][k].x +
						maxL1[i0][j2][l].y * kern2[0][2][l * 4 + 1][k].x +
						maxL1[i1][j0][l].y * kern2[1][0][l * 4 + 1][k].x +
						maxL1[i1][j1][l].y * kern2[1][1][l * 4 + 1][k].x +
						maxL1[i1][j2][l].y * kern2[1][2][l * 4 + 1][k].x +
						maxL1[i2][j0][l].y * kern2[2][0][l * 4 + 1][k].x +
						maxL1[i2][j1][l].y * kern2[2][1][l * 4 + 1][k].x +
						maxL1[i2][j2][l].y * kern2[2][2][l * 4 + 1][k].x;
					convL2[i][j][k].x +=
						maxL1[i0][j0][l].z * kern2[0][0][l * 4 + 2][k].x +
						maxL1[i0][j1][l].z * kern2[0][1][l * 4 + 2][k].x +
						maxL1[i0][j2][l].z * kern2[0][2][l * 4 + 2][k].x +
						maxL1[i1][j0][l].z * kern2[1][0][l * 4 + 2][k].x +
						maxL1[i1][j1][l].z * kern2[1][1][l * 4 + 2][k].x +
						maxL1[i1][j2][l].z * kern2[1][2][l * 4 + 2][k].x +
						maxL1[i2][j0][l].z * kern2[2][0][l * 4 + 2][k].x +
						maxL1[i2][j1][l].z * kern2[2][1][l * 4 + 2][k].x +
						maxL1[i2][j2][l].z * kern2[2][2][l * 4 + 2][k].x;
					convL2[i][j][k].x +=
						maxL1[i0][j0][l].w * kern2[0][0][l * 4 + 3][k].x +
						maxL1[i0][j1][l].w * kern2[0][1][l * 4 + 3][k].x +
						maxL1[i0][j2][l].w * kern2[0][2][l * 4 + 3][k].x +
						maxL1[i1][j0][l].w * kern2[1][0][l * 4 + 3][k].x +
						maxL1[i1][j1][l].w * kern2[1][1][l * 4 + 3][k].x +
						maxL1[i1][j2][l].w * kern2[1][2][l * 4 + 3][k].x +
						maxL1[i2][j0][l].w * kern2[2][0][l * 4 + 3][k].x +
						maxL1[i2][j1][l].w * kern2[2][1][l * 4 + 3][k].x +
						maxL1[i2][j2][l].w * kern2[2][2][l * 4 + 3][k].x;
					// Y
					convL2[i][j][k].y +=
						maxL1[i0][j0][l].x * kern2[0][0][l * 4][k].y +
						maxL1[i0][j1][l].x * kern2[0][1][l * 4][k].y +
						maxL1[i0][j2][l].x * kern2[0][2][l * 4][k].y +
						maxL1[i1][j0][l].x * kern2[1][0][l * 4][k].y +
						maxL1[i1][j1][l].x * kern2[1][1][l * 4][k].y +
						maxL1[i1][j2][l].x * kern2[1][2][l * 4][k].y +
						maxL1[i2][j0][l].x * kern2[2][0][l * 4][k].y +
						maxL1[i2][j1][l].x * kern2[2][1][l * 4][k].y +
						maxL1[i2][j2][l].x * kern2[2][2][l * 4][k].y;
					convL2[i][j][k].y +=
						maxL1[i0][j0][l].y * kern2[0][0][l * 4 + 1][k].y +
						maxL1[i0][j1][l].y * kern2[0][1][l * 4 + 1][k].y +
						maxL1[i0][j2][l].y * kern2[0][2][l * 4 + 1][k].y +
						maxL1[i1][j0][l].y * kern2[1][0][l * 4 + 1][k].y +
						maxL1[i1][j1][l].y * kern2[1][1][l * 4 + 1][k].y +
						maxL1[i1][j2][l].y * kern2[1][2][l * 4 + 1][k].y +
						maxL1[i2][j0][l].y * kern2[2][0][l * 4 + 1][k].y +
						maxL1[i2][j1][l].y * kern2[2][1][l * 4 + 1][k].y +
						maxL1[i2][j2][l].y * kern2[2][2][l * 4 + 1][k].y;
					convL2[i][j][k].y +=
						maxL1[i0][j0][l].z * kern2[0][0][l * 4 + 2][k].y +
						maxL1[i0][j1][l].z * kern2[0][1][l * 4 + 2][k].y +
						maxL1[i0][j2][l].z * kern2[0][2][l * 4 + 2][k].y +
						maxL1[i1][j0][l].z * kern2[1][0][l * 4 + 2][k].y +
						maxL1[i1][j1][l].z * kern2[1][1][l * 4 + 2][k].y +
						maxL1[i1][j2][l].z * kern2[1][2][l * 4 + 2][k].y +
						maxL1[i2][j0][l].z * kern2[2][0][l * 4 + 2][k].y +
						maxL1[i2][j1][l].z * kern2[2][1][l * 4 + 2][k].y +
						maxL1[i2][j2][l].z * kern2[2][2][l * 4 + 2][k].y;
					convL2[i][j][k].y +=
						maxL1[i0][j0][l].w * kern2[0][0][l * 4 + 3][k].y +
						maxL1[i0][j1][l].w * kern2[0][1][l * 4 + 3][k].y +
						maxL1[i0][j2][l].w * kern2[0][2][l * 4 + 3][k].y +
						maxL1[i1][j0][l].w * kern2[1][0][l * 4 + 3][k].y +
						maxL1[i1][j1][l].w * kern2[1][1][l * 4 + 3][k].y +
						maxL1[i1][j2][l].w * kern2[1][2][l * 4 + 3][k].y +
						maxL1[i2][j0][l].w * kern2[2][0][l * 4 + 3][k].y +
						maxL1[i2][j1][l].w * kern2[2][1][l * 4 + 3][k].y +
						maxL1[i2][j2][l].w * kern2[2][2][l * 4 + 3][k].y;
					// Z
					convL2[i][j][k].z +=
						maxL1[i0][j0][l].x * kern2[0][0][l * 4][k].z +
						maxL1[i0][j1][l].x * kern2[0][1][l * 4][k].z +
						maxL1[i0][j2][l].x * kern2[0][2][l * 4][k].z +
						maxL1[i1][j0][l].x * kern2[1][0][l * 4][k].z +
						maxL1[i1][j1][l].x * kern2[1][1][l * 4][k].z +
						maxL1[i1][j2][l].x * kern2[1][2][l * 4][k].z +
						maxL1[i2][j0][l].x * kern2[2][0][l * 4][k].z +
						maxL1[i2][j1][l].x * kern2[2][1][l * 4][k].z +
						maxL1[i2][j2][l].x * kern2[2][2][l * 4][k].z;
					convL2[i][j][k].z +=
						maxL1[i0][j0][l].y * kern2[0][0][l * 4 + 1][k].z +
						maxL1[i0][j1][l].y * kern2[0][1][l * 4 + 1][k].z +
						maxL1[i0][j2][l].y * kern2[0][2][l * 4 + 1][k].z +
						maxL1[i1][j0][l].y * kern2[1][0][l * 4 + 1][k].z +
						maxL1[i1][j1][l].y * kern2[1][1][l * 4 + 1][k].z +
						maxL1[i1][j2][l].y * kern2[1][2][l * 4 + 1][k].z +
						maxL1[i2][j0][l].y * kern2[2][0][l * 4 + 1][k].z +
						maxL1[i2][j1][l].y * kern2[2][1][l * 4 + 1][k].z +
						maxL1[i2][j2][l].y * kern2[2][2][l * 4 + 1][k].z;
					convL2[i][j][k].z +=
						maxL1[i0][j0][l].z * kern2[0][0][l * 4 + 2][k].z +
						maxL1[i0][j1][l].z * kern2[0][1][l * 4 + 2][k].z +
						maxL1[i0][j2][l].z * kern2[0][2][l * 4 + 2][k].z +
						maxL1[i1][j0][l].z * kern2[1][0][l * 4 + 2][k].z +
						maxL1[i1][j1][l].z * kern2[1][1][l * 4 + 2][k].z +
						maxL1[i1][j2][l].z * kern2[1][2][l * 4 + 2][k].z +
						maxL1[i2][j0][l].z * kern2[2][0][l * 4 + 2][k].z +
						maxL1[i2][j1][l].z * kern2[2][1][l * 4 + 2][k].z +
						maxL1[i2][j2][l].z * kern2[2][2][l * 4 + 2][k].z;
					convL2[i][j][k].z +=
						maxL1[i0][j0][l].w * kern2[0][0][l * 4 + 3][k].z +
						maxL1[i0][j1][l].w * kern2[0][1][l * 4 + 3][k].z +
						maxL1[i0][j2][l].w * kern2[0][2][l * 4 + 3][k].z +
						maxL1[i1][j0][l].w * kern2[1][0][l * 4 + 3][k].z +
						maxL1[i1][j1][l].w * kern2[1][1][l * 4 + 3][k].z +
						maxL1[i1][j2][l].w * kern2[1][2][l * 4 + 3][k].z +
						maxL1[i2][j0][l].w * kern2[2][0][l * 4 + 3][k].z +
						maxL1[i2][j1][l].w * kern2[2][1][l * 4 + 3][k].z +
						maxL1[i2][j2][l].w * kern2[2][2][l * 4 + 3][k].z;
					// W
					convL2[i][j][k].w +=
						maxL1[i0][j0][l].x * kern2[0][0][l * 4][k].w +
						maxL1[i0][j1][l].x * kern2[0][1][l * 4][k].w +
						maxL1[i0][j2][l].x * kern2[0][2][l * 4][k].w +
						maxL1[i1][j0][l].x * kern2[1][0][l * 4][k].w +
						maxL1[i1][j1][l].x * kern2[1][1][l * 4][k].w +
						maxL1[i1][j2][l].x * kern2[1][2][l * 4][k].w +
						maxL1[i2][j0][l].x * kern2[2][0][l * 4][k].w +
						maxL1[i2][j1][l].x * kern2[2][1][l * 4][k].w +
						maxL1[i2][j2][l].x * kern2[2][2][l * 4][k].w;
					convL2[i][j][k].w +=
						maxL1[i0][j0][l].y * kern2[0][0][l * 4 + 1][k].w +
						maxL1[i0][j1][l].y * kern2[0][1][l * 4 + 1][k].w +
						maxL1[i0][j2][l].y * kern2[0][2][l * 4 + 1][k].w +
						maxL1[i1][j0][l].y * kern2[1][0][l * 4 + 1][k].w +
						maxL1[i1][j1][l].y * kern2[1][1][l * 4 + 1][k].w +
						maxL1[i1][j2][l].y * kern2[1][2][l * 4 + 1][k].w +
						maxL1[i2][j0][l].y * kern2[2][0][l * 4 + 1][k].w +
						maxL1[i2][j1][l].y * kern2[2][1][l * 4 + 1][k].w +
						maxL1[i2][j2][l].y * kern2[2][2][l * 4 + 1][k].w;
					convL2[i][j][k].w +=
						maxL1[i0][j0][l].z * kern2[0][0][l * 4 + 2][k].w +
						maxL1[i0][j1][l].z * kern2[0][1][l * 4 + 2][k].w +
						maxL1[i0][j2][l].z * kern2[0][2][l * 4 + 2][k].w +
						maxL1[i1][j0][l].z * kern2[1][0][l * 4 + 2][k].w +
						maxL1[i1][j1][l].z * kern2[1][1][l * 4 + 2][k].w +
						maxL1[i1][j2][l].z * kern2[1][2][l * 4 + 2][k].w +
						maxL1[i2][j0][l].z * kern2[2][0][l * 4 + 2][k].w +
						maxL1[i2][j1][l].z * kern2[2][1][l * 4 + 2][k].w +
						maxL1[i2][j2][l].z * kern2[2][2][l * 4 + 2][k].w;
					convL2[i][j][k].w +=
						maxL1[i0][j0][l].w * kern2[0][0][l * 4 + 3][k].w +
						maxL1[i0][j1][l].w * kern2[0][1][l * 4 + 3][k].w +
						maxL1[i0][j2][l].w * kern2[0][2][l * 4 + 3][k].w +
						maxL1[i1][j0][l].w * kern2[1][0][l * 4 + 3][k].w +
						maxL1[i1][j1][l].w * kern2[1][1][l * 4 + 3][k].w +
						maxL1[i1][j2][l].w * kern2[1][2][l * 4 + 3][k].w +
						maxL1[i2][j0][l].w * kern2[2][0][l * 4 + 3][k].w +
						maxL1[i2][j1][l].w * kern2[2][1][l * 4 + 3][k].w +
						maxL1[i2][j2][l].w * kern2[2][2][l * 4 + 3][k].w;
				}

				// Bias
				convL2[i][j][k].x += bias2[k].x;
				convL2[i][j][k].y += bias2[k].y;
				convL2[i][j][k].z += bias2[k].z;
				convL2[i][j][k].w += bias2[k].w;

				// Activation
				convL2[i][j][k].x = fnELU(convL2[i][j][k].x, 0.15f);
				convL2[i][j][k].y = fnELU(convL2[i][j][k].y, 0.15f);
				convL2[i][j][k].z = fnELU(convL2[i][j][k].z, 0.15f);
				convL2[i][j][k].w = fnELU(convL2[i][j][k].w, 0.15f);
			}
		}
	}

	// Max pooling layer 2, size=2x2, stride=2
	for (int k = 0; k < 16; k++) {
		for (int i = 0; i < 7; i++) {
			for (int j = 0; j < 7; j++) {
				int i0 = i * 2, i1 = i0 + 1;
				int j0 = j * 2, j1 = j0 + 1;

				float m = convL2[i0][j0][k].x;
				m = fmaxf(m, convL2[i0][j1][k].x);
				m = fmaxf(m, convL2[i1][j0][k].x);
				m = fmaxf(m, convL2[i1][j1][k].x);
				maxL2[i][j][k].x = m;

				m = convL2[i0][j0][k].y;
				m = fmaxf(m, convL2[i0][j1][k].y);
				m = fmaxf(m, convL2[i1][j0][k].y);
				m = fmaxf(m, convL2[i1][j1][k].y);
				maxL2[i][j][k].y = m;

				m = convL2[i0][j0][k].z;
				m = fmaxf(m, convL2[i0][j1][k].z);
				m = fmaxf(m, convL2[i1][j0][k].z);
				m = fmaxf(m, convL2[i1][j1][k].z);
				maxL2[i][j][k].z = m;

				m = convL2[i0][j0][k].w;
				m = fmaxf(m, convL2[i0][j1][k].w);
				m = fmaxf(m, convL2[i1][j0][k].w);
				m = fmaxf(m, convL2[i1][j1][k].w);
				maxL2[i][j][k].w = m;
			}
		}
	}

	// Max pooling layer 2 index
	for (int k = 0; k < 16; k++) {
		for (int i = 0; i < 7; i++) {
			for (int j = 0; j < 7; j++) {
				int i0 = i * 2, i1 = i0 + 1;
				int j0 = j * 2, j1 = j0 + 1;

				// X
				float m = convL2[i0][j0][k].x;
				imaxL2[i][j][k].x = i0 * 14 + j0;
				m = fmaxf(m, convL2[i0][j1][k].x);
				imaxL2[i][j][k].x = (m == convL2[i0][j1][k].x) ? 
					(i0 * 14 + j1) : imaxL2[i][j][k].x;
				m = fmaxf(m, convL2[i1][j0][k].x);
				imaxL2[i][j][k].x = (m == convL2[i1][j0][k].x) ?
					(i1 * 14 + j0) : imaxL2[i][j][k].x;
				m = fmaxf(m, convL2[i1][j1][k].x);
				imaxL2[i][j][k].x = (m == convL2[i1][j1][k].x) ?
					(i1 * 14 + j1) : imaxL2[i][j][k].x;

				// Y
				m = convL2[i0][j0][k].y;
				imaxL2[i][j][k].y = i0 * 14 + j0;
				m = fmaxf(m, convL2[i0][j1][k].y);
				imaxL2[i][j][k].y = (m == convL2[i0][j1][k].y) ?
					(i0 * 14 + j1) : imaxL2[i][j][k].y;
				m = fmaxf(m, convL2[i1][j0][k].y);
				imaxL2[i][j][k].y = (m == convL2[i1][j0][k].y) ?
					(i1 * 14 + j0) : imaxL2[i][j][k].y;
				m = fmaxf(m, convL2[i1][j1][k].y);
				imaxL2[i][j][k].y = (m == convL2[i1][j1][k].y) ?
					(i1 * 14 + j1) : imaxL2[i][j][k].y;

				// Z
				m = convL2[i0][j0][k].z;
				imaxL2[i][j][k].z = i0 * 14 + j0;
				m = fmaxf(m, convL2[i0][j1][k].z);
				imaxL2[i][j][k].z = (m == convL2[i0][j1][k].z) ?
					(i0 * 14 + j1) : imaxL2[i][j][k].z;
				m = fmaxf(m, convL2[i1][j0][k].z);
				imaxL2[i][j][k].z = (m == convL2[i1][j0][k].z) ?
					(i1 * 14 + j0) : imaxL2[i][j][k].z;
				m = fmaxf(m, convL2[i1][j1][k].z);
				imaxL2[i][j][k].z = (m == convL2[i1][j1][k].z) ?
					(i1 * 14 + j1) : imaxL2[i][j][k].z;

				// W
				m = convL2[i0][j0][k].w;
				imaxL2[i][j][k].w = i0 * 14 + j0;
				m = fmaxf(m, convL2[i0][j1][k].w);
				imaxL2[i][j][k].w = (m == convL2[i0][j1][k].w) ?
					(i0 * 14 + j1) : imaxL2[i][j][k].w;
				m = fmaxf(m, convL2[i1][j0][k].w);
				imaxL2[i][j][k].w = (m == convL2[i1][j0][k].w) ?
					(i1 * 14 + j0) : imaxL2[i][j][k].w;
				m = fmaxf(m, convL2[i1][j1][k].w);
				imaxL2[i][j][k].w = (m == convL2[i1][j1][k].w) ?
					(i1 * 14 + j1) : imaxL2[i][j][k].w;
			}
		}
	}

	// Convolutional layer 3, kernel=3x3, pad=1, stride=2
	for (int k = 0; k < 32; k++) {
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				int i0 = i * 2, i1 = i0 - 1, i2 = i0 + 1;
				int j0 = j * 2, j1 = j0 - 1, j2 = j0 + 1;
				
				// Padding
				bool bi1 = i1 < 0, bj1 = j1 < 0, bi2 = i2 > 6, bj2 = j2 > 6;
				bool b02 = bi1 || bj1, b03 = bi1 || bj2, b04 = bi2 || bj1, b05 = bi2 || bj2;

				for (int l = 0; l < 16; l++) {
					// X
					convL3[i][j][k].x +=
						(b02 ? 0.0f : maxL2[i1][j1][l].x * kern3[0][0][l * 4][k].x) +
						(bi1 ? 0.0f : maxL2[i1][j0][l].x * kern3[0][1][l * 4][k].x) +
						(b03 ? 0.0f : maxL2[i1][j2][l].x * kern3[0][2][l * 4][k].x) +
						(bj1 ? 0.0f : maxL2[i0][j1][l].x * kern3[1][0][l * 4][k].x) +
						maxL2[i0][j0][l].x * kern3[1][1][l * 4][k].x +
						(bj2 ? 0.0f : maxL2[i0][j2][l].x * kern3[1][2][l * 4][k].x) +
						(b04 ? 0.0f : maxL2[i2][j1][l].x * kern3[2][0][l * 4][k].x) +
						(bi2 ? 0.0f : maxL2[i2][j0][l].x * kern3[2][1][l * 4][k].x) +
						(b05 ? 0.0f : maxL2[i2][j2][l].x * kern3[2][2][l * 4][k].x);
					convL3[i][j][k].x +=
						(b02 ? 0.0f : maxL2[i1][j1][l].y * kern3[0][0][l * 4 + 1][k].x) +
						(bi1 ? 0.0f : maxL2[i1][j0][l].y * kern3[0][1][l * 4 + 1][k].x) +
						(b03 ? 0.0f : maxL2[i1][j2][l].y * kern3[0][2][l * 4 + 1][k].x) +
						(bj1 ? 0.0f : maxL2[i0][j1][l].y * kern3[1][0][l * 4 + 1][k].x) +
						maxL2[i0][j0][l].y * kern3[1][1][l * 4 + 1][k].x +
						(bj2 ? 0.0f : maxL2[i0][j2][l].y * kern3[1][2][l * 4 + 1][k].x) +
						(b04 ? 0.0f : maxL2[i2][j1][l].y * kern3[2][0][l * 4 + 1][k].x) +
						(bi2 ? 0.0f : maxL2[i2][j0][l].y * kern3[2][1][l * 4 + 1][k].x) +
						(b05 ? 0.0f : maxL2[i2][j2][l].y * kern3[2][2][l * 4 + 1][k].x);
					convL3[i][j][k].x +=
						(b02 ? 0.0f : maxL2[i1][j1][l].z * kern3[0][0][l * 4 + 2][k].x) +
						(bi1 ? 0.0f : maxL2[i1][j0][l].z * kern3[0][1][l * 4 + 2][k].x) +
						(b03 ? 0.0f : maxL2[i1][j2][l].z * kern3[0][2][l * 4 + 2][k].x) +
						(bj1 ? 0.0f : maxL2[i0][j1][l].z * kern3[1][0][l * 4 + 2][k].x) +
						maxL2[i0][j0][l].z * kern3[1][1][l * 4 + 2][k].x +
						(bj2 ? 0.0f : maxL2[i0][j2][l].z * kern3[1][2][l * 4 + 2][k].x) +
						(b04 ? 0.0f : maxL2[i2][j1][l].z * kern3[2][0][l * 4 + 2][k].x) +
						(bi2 ? 0.0f : maxL2[i2][j0][l].z * kern3[2][1][l * 4 + 2][k].x) +
						(b05 ? 0.0f : maxL2[i2][j2][l].z * kern3[2][2][l * 4 + 2][k].x);
					convL3[i][j][k].x +=
						(b02 ? 0.0f : maxL2[i1][j1][l].w * kern3[0][0][l * 4 + 3][k].x) +
						(bi1 ? 0.0f : maxL2[i1][j0][l].w * kern3[0][1][l * 4 + 3][k].x) +
						(b03 ? 0.0f : maxL2[i1][j2][l].w * kern3[0][2][l * 4 + 3][k].x) +
						(bj1 ? 0.0f : maxL2[i0][j1][l].w * kern3[1][0][l * 4 + 3][k].x) +
						maxL2[i0][j0][l].w * kern3[1][1][l * 4 + 3][k].x +
						(bj2 ? 0.0f : maxL2[i0][j2][l].w * kern3[1][2][l * 4 + 3][k].x) +
						(b04 ? 0.0f : maxL2[i2][j1][l].w * kern3[2][0][l * 4 + 3][k].x) +
						(bi2 ? 0.0f : maxL2[i2][j0][l].w * kern3[2][1][l * 4 + 3][k].x) +
						(b05 ? 0.0f : maxL2[i2][j2][l].w * kern3[2][2][l * 4 + 3][k].x);
					// Y
					convL3[i][j][k].y +=
						(b02 ? 0.0f : maxL2[i1][j1][l].x * kern3[0][0][l * 4][k].y) +
						(bi1 ? 0.0f : maxL2[i1][j0][l].x * kern3[0][1][l * 4][k].y) +
						(b03 ? 0.0f : maxL2[i1][j2][l].x * kern3[0][2][l * 4][k].y) +
						(bj1 ? 0.0f : maxL2[i0][j1][l].x * kern3[1][0][l * 4][k].y) +
						maxL2[i0][j0][l].x * kern3[1][1][l * 4][k].y +
						(bj2 ? 0.0f : maxL2[i0][j2][l].x * kern3[1][2][l * 4][k].y) +
						(b04 ? 0.0f : maxL2[i2][j1][l].x * kern3[2][0][l * 4][k].y) +
						(bi2 ? 0.0f : maxL2[i2][j0][l].x * kern3[2][1][l * 4][k].y) +
						(b05 ? 0.0f : maxL2[i2][j2][l].x * kern3[2][2][l * 4][k].y);
					convL3[i][j][k].y +=
						(b02 ? 0.0f : maxL2[i1][j1][l].y * kern3[0][0][l * 4 + 1][k].y) +
						(bi1 ? 0.0f : maxL2[i1][j0][l].y * kern3[0][1][l * 4 + 1][k].y) +
						(b03 ? 0.0f : maxL2[i1][j2][l].y * kern3[0][2][l * 4 + 1][k].y) +
						(bj1 ? 0.0f : maxL2[i0][j1][l].y * kern3[1][0][l * 4 + 1][k].y) +
						maxL2[i0][j0][l].y * kern3[1][1][l * 4 + 1][k].y +
						(bj2 ? 0.0f : maxL2[i0][j2][l].y * kern3[1][2][l * 4 + 1][k].y) +
						(b04 ? 0.0f : maxL2[i2][j1][l].y * kern3[2][0][l * 4 + 1][k].y) +
						(bi2 ? 0.0f : maxL2[i2][j0][l].y * kern3[2][1][l * 4 + 1][k].y) +
						(b05 ? 0.0f : maxL2[i2][j2][l].y * kern3[2][2][l * 4 + 1][k].y);
					convL3[i][j][k].y +=
						(b02 ? 0.0f : maxL2[i1][j1][l].z * kern3[0][0][l * 4 + 2][k].y) +
						(bi1 ? 0.0f : maxL2[i1][j0][l].z * kern3[0][1][l * 4 + 2][k].y) +
						(b03 ? 0.0f : maxL2[i1][j2][l].z * kern3[0][2][l * 4 + 2][k].y) +
						(bj1 ? 0.0f : maxL2[i0][j1][l].z * kern3[1][0][l * 4 + 2][k].y) +
						maxL2[i0][j0][l].z * kern3[1][1][l * 4 + 2][k].y +
						(bj2 ? 0.0f : maxL2[i0][j2][l].z * kern3[1][2][l * 4 + 2][k].y) +
						(b04 ? 0.0f : maxL2[i2][j1][l].z * kern3[2][0][l * 4 + 2][k].y) +
						(bi2 ? 0.0f : maxL2[i2][j0][l].z * kern3[2][1][l * 4 + 2][k].y) +
						(b05 ? 0.0f : maxL2[i2][j2][l].z * kern3[2][2][l * 4 + 2][k].y);
					convL3[i][j][k].y +=
						(b02 ? 0.0f : maxL2[i1][j1][l].w * kern3[0][0][l * 4 + 3][k].y) +
						(bi1 ? 0.0f : maxL2[i1][j0][l].w * kern3[0][1][l * 4 + 3][k].y) +
						(b03 ? 0.0f : maxL2[i1][j2][l].w * kern3[0][2][l * 4 + 3][k].y) +
						(bj1 ? 0.0f : maxL2[i0][j1][l].w * kern3[1][0][l * 4 + 3][k].y) +
						maxL2[i0][j0][l].w * kern3[1][1][l * 4 + 3][k].y +
						(bj2 ? 0.0f : maxL2[i0][j2][l].w * kern3[1][2][l * 4 + 3][k].y) +
						(b04 ? 0.0f : maxL2[i2][j1][l].w * kern3[2][0][l * 4 + 3][k].y) +
						(bi2 ? 0.0f : maxL2[i2][j0][l].w * kern3[2][1][l * 4 + 3][k].y) +
						(b05 ? 0.0f : maxL2[i2][j2][l].w * kern3[2][2][l * 4 + 3][k].y);
					// Z
					convL3[i][j][k].z +=
						(b02 ? 0.0f : maxL2[i1][j1][l].x * kern3[0][0][l * 4][k].z) +
						(bi1 ? 0.0f : maxL2[i1][j0][l].x * kern3[0][1][l * 4][k].z) +
						(b03 ? 0.0f : maxL2[i1][j2][l].x * kern3[0][2][l * 4][k].z) +
						(bj1 ? 0.0f : maxL2[i0][j1][l].x * kern3[1][0][l * 4][k].z) +
						maxL2[i0][j0][l].x * kern3[1][1][l * 4][k].z +
						(bj2 ? 0.0f : maxL2[i0][j2][l].x * kern3[1][2][l * 4][k].z) +
						(b04 ? 0.0f : maxL2[i2][j1][l].x * kern3[2][0][l * 4][k].z) +
						(bi2 ? 0.0f : maxL2[i2][j0][l].x * kern3[2][1][l * 4][k].z) +
						(b05 ? 0.0f : maxL2[i2][j2][l].x * kern3[2][2][l * 4][k].z);
					convL3[i][j][k].z +=
						(b02 ? 0.0f : maxL2[i1][j1][l].y * kern3[0][0][l * 4 + 1][k].z) +
						(bi1 ? 0.0f : maxL2[i1][j0][l].y * kern3[0][1][l * 4 + 1][k].z) +
						(b03 ? 0.0f : maxL2[i1][j2][l].y * kern3[0][2][l * 4 + 1][k].z) +
						(bj1 ? 0.0f : maxL2[i0][j1][l].y * kern3[1][0][l * 4 + 1][k].z) +
						maxL2[i0][j0][l].y * kern3[1][1][l * 4 + 1][k].z +
						(bj2 ? 0.0f : maxL2[i0][j2][l].y * kern3[1][2][l * 4 + 1][k].z) +
						(b04 ? 0.0f : maxL2[i2][j1][l].y * kern3[2][0][l * 4 + 1][k].z) +
						(bi2 ? 0.0f : maxL2[i2][j0][l].y * kern3[2][1][l * 4 + 1][k].z) +
						(b05 ? 0.0f : maxL2[i2][j2][l].y * kern3[2][2][l * 4 + 1][k].z);
					convL3[i][j][k].z +=
						(b02 ? 0.0f : maxL2[i1][j1][l].z * kern3[0][0][l * 4 + 2][k].z) +
						(bi1 ? 0.0f : maxL2[i1][j0][l].z * kern3[0][1][l * 4 + 2][k].z) +
						(b03 ? 0.0f : maxL2[i1][j2][l].z * kern3[0][2][l * 4 + 2][k].z) +
						(bj1 ? 0.0f : maxL2[i0][j1][l].z * kern3[1][0][l * 4 + 2][k].z) +
						maxL2[i0][j0][l].z * kern3[1][1][l * 4 + 2][k].z +
						(bj2 ? 0.0f : maxL2[i0][j2][l].z * kern3[1][2][l * 4 + 2][k].z) +
						(b04 ? 0.0f : maxL2[i2][j1][l].z * kern3[2][0][l * 4 + 2][k].z) +
						(bi2 ? 0.0f : maxL2[i2][j0][l].z * kern3[2][1][l * 4 + 2][k].z) +
						(b05 ? 0.0f : maxL2[i2][j2][l].z * kern3[2][2][l * 4 + 2][k].z);
					convL3[i][j][k].z +=
						(b02 ? 0.0f : maxL2[i1][j1][l].w * kern3[0][0][l * 4 + 3][k].z) +
						(bi1 ? 0.0f : maxL2[i1][j0][l].w * kern3[0][1][l * 4 + 3][k].z) +
						(b03 ? 0.0f : maxL2[i1][j2][l].w * kern3[0][2][l * 4 + 3][k].z) +
						(bj1 ? 0.0f : maxL2[i0][j1][l].w * kern3[1][0][l * 4 + 3][k].z) +
						maxL2[i0][j0][l].w * kern3[1][1][l * 4 + 3][k].z +
						(bj2 ? 0.0f : maxL2[i0][j2][l].w * kern3[1][2][l * 4 + 3][k].z) +
						(b04 ? 0.0f : maxL2[i2][j1][l].w * kern3[2][0][l * 4 + 3][k].z) +
						(bi2 ? 0.0f : maxL2[i2][j0][l].w * kern3[2][1][l * 4 + 3][k].z) +
						(b05 ? 0.0f : maxL2[i2][j2][l].w * kern3[2][2][l * 4 + 3][k].z);
					// W
					convL3[i][j][k].w +=
						(b02 ? 0.0f : maxL2[i1][j1][l].x * kern3[0][0][l * 4][k].w) +
						(bi1 ? 0.0f : maxL2[i1][j0][l].x * kern3[0][1][l * 4][k].w) +
						(b03 ? 0.0f : maxL2[i1][j2][l].x * kern3[0][2][l * 4][k].w) +
						(bj1 ? 0.0f : maxL2[i0][j1][l].x * kern3[1][0][l * 4][k].w) +
						maxL2[i0][j0][l].x * kern3[1][1][l * 4][k].w +
						(bj2 ? 0.0f : maxL2[i0][j2][l].x * kern3[1][2][l * 4][k].w) +
						(b04 ? 0.0f : maxL2[i2][j1][l].x * kern3[2][0][l * 4][k].w) +
						(bi2 ? 0.0f : maxL2[i2][j0][l].x * kern3[2][1][l * 4][k].w) +
						(b05 ? 0.0f : maxL2[i2][j2][l].x * kern3[2][2][l * 4][k].w);
					convL3[i][j][k].w +=
						(b02 ? 0.0f : maxL2[i1][j1][l].y * kern3[0][0][l * 4 + 1][k].w) +
						(bi1 ? 0.0f : maxL2[i1][j0][l].y * kern3[0][1][l * 4 + 1][k].w) +
						(b03 ? 0.0f : maxL2[i1][j2][l].y * kern3[0][2][l * 4 + 1][k].w) +
						(bj1 ? 0.0f : maxL2[i0][j1][l].y * kern3[1][0][l * 4 + 1][k].w) +
						maxL2[i0][j0][l].y * kern3[1][1][l * 4 + 1][k].w +
						(bj2 ? 0.0f : maxL2[i0][j2][l].y * kern3[1][2][l * 4 + 1][k].w) +
						(b04 ? 0.0f : maxL2[i2][j1][l].y * kern3[2][0][l * 4 + 1][k].w) +
						(bi2 ? 0.0f : maxL2[i2][j0][l].y * kern3[2][1][l * 4 + 1][k].w) +
						(b05 ? 0.0f : maxL2[i2][j2][l].y * kern3[2][2][l * 4 + 1][k].w);
					convL3[i][j][k].w +=
						(b02 ? 0.0f : maxL2[i1][j1][l].z * kern3[0][0][l * 4 + 2][k].w) +
						(bi1 ? 0.0f : maxL2[i1][j0][l].z * kern3[0][1][l * 4 + 2][k].w) +
						(b03 ? 0.0f : maxL2[i1][j2][l].z * kern3[0][2][l * 4 + 2][k].w) +
						(bj1 ? 0.0f : maxL2[i0][j1][l].z * kern3[1][0][l * 4 + 2][k].w) +
						maxL2[i0][j0][l].z * kern3[1][1][l * 4 + 2][k].w +
						(bj2 ? 0.0f : maxL2[i0][j2][l].z * kern3[1][2][l * 4 + 2][k].w) +
						(b04 ? 0.0f : maxL2[i2][j1][l].z * kern3[2][0][l * 4 + 2][k].w) +
						(bi2 ? 0.0f : maxL2[i2][j0][l].z * kern3[2][1][l * 4 + 2][k].w) +
						(b05 ? 0.0f : maxL2[i2][j2][l].z * kern3[2][2][l * 4 + 2][k].w);
					convL3[i][j][k].w +=
						(b02 ? 0.0f : maxL2[i1][j1][l].w * kern3[0][0][l * 4 + 3][k].w) +
						(bi1 ? 0.0f : maxL2[i1][j0][l].w * kern3[0][1][l * 4 + 3][k].w) +
						(b03 ? 0.0f : maxL2[i1][j2][l].w * kern3[0][2][l * 4 + 3][k].w) +
						(bj1 ? 0.0f : maxL2[i0][j1][l].w * kern3[1][0][l * 4 + 3][k].w) +
						maxL2[i0][j0][l].w * kern3[1][1][l * 4 + 3][k].w +
						(bj2 ? 0.0f : maxL2[i0][j2][l].w * kern3[1][2][l * 4 + 3][k].w) +
						(b04 ? 0.0f : maxL2[i2][j1][l].w * kern3[2][0][l * 4 + 3][k].w) +
						(bi2 ? 0.0f : maxL2[i2][j0][l].w * kern3[2][1][l * 4 + 3][k].w) +
						(b05 ? 0.0f : maxL2[i2][j2][l].w * kern3[2][2][l * 4 + 3][k].w);
				}

				// Bias
				convL3[i][j][k].x += bias3[k].x;
				convL3[i][j][k].y += bias3[k].y;
				convL3[i][j][k].z += bias3[k].z;
				convL3[i][j][k].w += bias3[k].w;

				// Activation
				convL3[i][j][k].x = fnELU(convL3[i][j][k].x, 0.15f);
				convL3[i][j][k].y = fnELU(convL3[i][j][k].y, 0.15f);
				convL3[i][j][k].z = fnELU(convL3[i][j][k].z, 0.15f);
				convL3[i][j][k].w = fnELU(convL3[i][j][k].w, 0.15f);
			}
		}
	}

	// Max pooling layer 3, size=2x2, stride=2
	for (int k = 0; k < 32; k++) {
		// X
		float m = convL3[0][0][k].x;
		m = fmaxf(m, convL3[0][1][k].x);
		m = fmaxf(m, convL3[1][0][k].x);
		m = fmaxf(m, convL3[1][1][k].x);
		maxL3[k * 4].x = m;
		m = convL3[0][2][k].x;
		m = fmaxf(m, convL3[0][3][k].x);
		m = fmaxf(m, convL3[1][2][k].x);
		m = fmaxf(m, convL3[1][3][k].x);
		maxL3[k * 4].y = m;
		m = convL3[2][0][k].x;
		m = fmaxf(m, convL3[2][1][k].x);
		m = fmaxf(m, convL3[3][0][k].x);
		m = fmaxf(m, convL3[3][1][k].x);
		maxL3[k * 4].z = m;
		m = convL3[2][2][k].x;
		m = fmaxf(m, convL3[2][3][k].x);
		m = fmaxf(m, convL3[3][2][k].x);
		m = fmaxf(m, convL3[3][3][k].x);
		maxL3[k * 4].w = m;
		// Y
		m = convL3[0][0][k].y;
		m = fmaxf(m, convL3[0][1][k].y);
		m = fmaxf(m, convL3[1][0][k].y);
		m = fmaxf(m, convL3[1][1][k].y);
		maxL3[k * 4 + 1].x = m;
		m = convL3[0][2][k].y;
		m = fmaxf(m, convL3[0][3][k].y);
		m = fmaxf(m, convL3[1][2][k].y);
		m = fmaxf(m, convL3[1][3][k].y);
		maxL3[k * 4 + 1].y = m;
		m = convL3[2][0][k].y;
		m = fmaxf(m, convL3[2][1][k].y);
		m = fmaxf(m, convL3[3][0][k].y);
		m = fmaxf(m, convL3[3][1][k].y);
		maxL3[k * 4 + 1].z = m;
		m = convL3[2][2][k].y;
		m = fmaxf(m, convL3[2][3][k].y);
		m = fmaxf(m, convL3[3][2][k].y);
		m = fmaxf(m, convL3[3][3][k].y);
		maxL3[k * 4 + 1].w = m;
		// Z
		m = convL3[0][0][k].z;
		m = fmaxf(m, convL3[0][1][k].z);
		m = fmaxf(m, convL3[1][0][k].z);
		m = fmaxf(m, convL3[1][1][k].z);
		maxL3[k * 4 + 2].x = m;
		m = convL3[0][2][k].z;
		m = fmaxf(m, convL3[0][3][k].z);
		m = fmaxf(m, convL3[1][2][k].z);
		m = fmaxf(m, convL3[1][3][k].z);
		maxL3[k * 4 + 2].y = m;
		m = convL3[2][0][k].z;
		m = fmaxf(m, convL3[2][1][k].z);
		m = fmaxf(m, convL3[3][0][k].z);
		m = fmaxf(m, convL3[3][1][k].z);
		maxL3[k * 4 + 2].z = m;
		m = convL3[2][2][k].z;
		m = fmaxf(m, convL3[2][3][k].z);
		m = fmaxf(m, convL3[3][2][k].z);
		m = fmaxf(m, convL3[3][3][k].z);
		maxL3[k * 4 + 2].w = m;
		// W
		m = convL3[0][0][k].w;
		m = fmaxf(m, convL3[0][1][k].w);
		m = fmaxf(m, convL3[1][0][k].w);
		m = fmaxf(m, convL3[1][1][k].w);
		maxL3[k * 4 + 3].x = m;
		m = convL3[0][2][k].w;
		m = fmaxf(m, convL3[0][3][k].w);
		m = fmaxf(m, convL3[1][2][k].w);
		m = fmaxf(m, convL3[1][3][k].w);
		maxL3[k * 4 + 3].y = m;
		m = convL3[2][0][k].w;
		m = fmaxf(m, convL3[2][1][k].w);
		m = fmaxf(m, convL3[3][0][k].w);
		m = fmaxf(m, convL3[3][1][k].w);
		maxL3[k * 4 + 3].z = m;
		m = convL3[2][2][k].w;
		m = fmaxf(m, convL3[2][3][k].w);
		m = fmaxf(m, convL3[3][2][k].w);
		m = fmaxf(m, convL3[3][3][k].w);
		maxL3[k * 4 + 3].w = m;
	}

	// Max pooling layer 3 index
	for (int k = 0; k < 32; k++) {
		// X
		float m = convL3[0][0][k].x;
		imaxL3[k * 4].x = 0;
		m = fmaxf(m, convL3[0][1][k].x);
		imaxL3[k * 4].x = (m == convL3[0][1][k].x) ? 1 : imaxL3[k * 4].x;
		m = fmaxf(m, convL3[1][0][k].x);
		imaxL3[k * 4].x = (m == convL3[1][0][k].x) ? 4 : imaxL3[k * 4].x;
		m = fmaxf(m, convL3[1][1][k].x);
		imaxL3[k * 4].x = (m == convL3[1][1][k].x) ? 5 : imaxL3[k * 4].x;

		m = convL3[0][2][k].x;
		imaxL3[k * 4].y = 2;
		m = fmaxf(m, convL3[0][3][k].x);
		imaxL3[k * 4].y = (m == convL3[0][3][k].x) ? 3 : imaxL3[k * 4].y;
		m = fmaxf(m, convL3[1][2][k].x);
		imaxL3[k * 4].y = (m == convL3[1][2][k].x) ? 6 : imaxL3[k * 4].y;
		m = fmaxf(m, convL3[1][3][k].x);
		imaxL3[k * 4].y = (m == convL3[1][3][k].x) ? 7 : imaxL3[k * 4].y;

		m = convL3[2][0][k].x;
		imaxL3[k * 4].z = 8;
		m = fmaxf(m, convL3[2][1][k].x);
		imaxL3[k * 4].z = (m == convL3[2][1][k].x) ? 9 : imaxL3[k * 4].z;
		m = fmaxf(m, convL3[3][0][k].x);
		imaxL3[k * 4].z = (m == convL3[3][0][k].x) ? 12 : imaxL3[k * 4].z;
		m = fmaxf(m, convL3[3][1][k].x);
		imaxL3[k * 4].z = (m == convL3[3][1][k].x) ? 13 : imaxL3[k * 4].z;

		m = convL3[2][2][k].x;
		imaxL3[k * 4].w = 10;
		m = fmaxf(m, convL3[2][3][k].x);
		imaxL3[k * 4].w = (m == convL3[2][3][k].x) ? 11 : imaxL3[k * 4].w;
		m = fmaxf(m, convL3[3][2][k].x);
		imaxL3[k * 4].w = (m == convL3[3][2][k].x) ? 14 : imaxL3[k * 4].w;
		m = fmaxf(m, convL3[3][3][k].x);
		imaxL3[k * 4].w = (m == convL3[3][3][k].x) ? 15 : imaxL3[k * 4].w;

		// Y
		m = convL3[0][0][k].y;
		m = fmaxf(m, convL3[0][1][k].y);
		m = fmaxf(m, convL3[1][0][k].y);
		m = fmaxf(m, convL3[1][1][k].y);
		maxL3[k * 4 + 1].x = m;
		m = convL3[0][2][k].y;
		m = fmaxf(m, convL3[0][3][k].y);
		m = fmaxf(m, convL3[1][2][k].y);
		m = fmaxf(m, convL3[1][3][k].y);
		maxL3[k * 4 + 1].y = m;
		m = convL3[2][0][k].y;
		m = fmaxf(m, convL3[2][1][k].y);
		m = fmaxf(m, convL3[3][0][k].y);
		m = fmaxf(m, convL3[3][1][k].y);
		maxL3[k * 4 + 1].z = m;
		m = convL3[2][2][k].y;
		m = fmaxf(m, convL3[2][3][k].y);
		m = fmaxf(m, convL3[3][2][k].y);
		m = fmaxf(m, convL3[3][3][k].y);
		maxL3[k * 4 + 1].w = m;
		// Z
		m = convL3[0][0][k].z;
		m = fmaxf(m, convL3[0][1][k].z);
		m = fmaxf(m, convL3[1][0][k].z);
		m = fmaxf(m, convL3[1][1][k].z);
		maxL3[k * 4 + 2].x = m;
		m = convL3[0][2][k].z;
		m = fmaxf(m, convL3[0][3][k].z);
		m = fmaxf(m, convL3[1][2][k].z);
		m = fmaxf(m, convL3[1][3][k].z);
		maxL3[k * 4 + 2].y = m;
		m = convL3[2][0][k].z;
		m = fmaxf(m, convL3[2][1][k].z);
		m = fmaxf(m, convL3[3][0][k].z);
		m = fmaxf(m, convL3[3][1][k].z);
		maxL3[k * 4 + 2].z = m;
		m = convL3[2][2][k].z;
		m = fmaxf(m, convL3[2][3][k].z);
		m = fmaxf(m, convL3[3][2][k].z);
		m = fmaxf(m, convL3[3][3][k].z);
		maxL3[k * 4 + 2].w = m;
		// W
		m = convL3[0][0][k].w;
		m = fmaxf(m, convL3[0][1][k].w);
		m = fmaxf(m, convL3[1][0][k].w);
		m = fmaxf(m, convL3[1][1][k].w);
		maxL3[k * 4 + 3].x = m;
		m = convL3[0][2][k].w;
		m = fmaxf(m, convL3[0][3][k].w);
		m = fmaxf(m, convL3[1][2][k].w);
		m = fmaxf(m, convL3[1][3][k].w);
		maxL3[k * 4 + 3].y = m;
		m = convL3[2][0][k].w;
		m = fmaxf(m, convL3[2][1][k].w);
		m = fmaxf(m, convL3[3][0][k].w);
		m = fmaxf(m, convL3[3][1][k].w);
		maxL3[k * 4 + 3].z = m;
		m = convL3[2][2][k].w;
		m = fmaxf(m, convL3[2][3][k].w);
		m = fmaxf(m, convL3[3][2][k].w);
		m = fmaxf(m, convL3[3][3][k].w);
		maxL3[k * 4 + 3].w = m;
	}

	// FC1
	for (int i = 0; i < 32; i++) {
		fc1[i].x = 0;
	}

	auto t2 = chrono::high_resolution_clock::now();

	// Print debugging

	string out;
	out += "kern1 weights\n";
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				out += to_string(kern1[i][j][k][0].x);
				out.push_back(' ');
			}
			out.push_back('\n');
		}
	}

	out += "kern1 weights\n";
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				out += to_string(kern1[i][j][k][7].w);
				out.push_back(' ');
			}
			out.push_back('\n');
		}
	}

	out += "\nconv1\n";
	for (int i = 0; i < 32; i++) {
		for (int j = 0; j < 32; j++) {
			out += to_string(convL1[i][j][0].x);
			out.push_back(' ');
		}
		out.push_back('\n');
	}

	out += "\nconv1\n";
	for (int i = 0; i < 32; i++) {
		for (int j = 0; j < 32; j++) {
			out += to_string(convL1[i][j][7].w);
			out.push_back(' ');
		}
		out.push_back('\n');
	}

	out += "\nmax1\n";
	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++) {
			out += to_string(maxL1[i][j][0].x);
			out.push_back(' ');
		}
		out.push_back('\n');
	}

	out += "\nmax1\n";
	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++) {
			out += to_string(maxL1[i][j][7].w);
			out.push_back(' ');
		}
		out.push_back('\n');
	}

	out += "\nmax1 index\n";
	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++) {
			out += to_string(imaxL1[i][j][0].x);
			out.push_back(' ');
		}
		out.push_back('\n');
	}

	out += "\nmax1 index\n";
	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++) {
			out += to_string(imaxL1[i][j][7].w);
			out.push_back(' ');
		}
		out.push_back('\n');
	}

	out += "\nconv2\n";
	for (int i = 0; i < 14; i++) {
		for (int j = 0; j < 14; j++) {
			out += to_string(convL2[i][j][0].x);
			out.push_back(' ');
		}
		out.push_back('\n');
	}

	out += "\nconv2\n";
	for (int i = 0; i < 14; i++) {
		for (int j = 0; j < 14; j++) {
			out += to_string(convL2[i][j][15].w);
			out.push_back(' ');
		}
		out.push_back('\n');
	}

	out += "\nmax2\n";
	for (int i = 0; i < 7; i++) {
		for (int j = 0; j < 7; j++) {
			out += to_string(maxL2[i][j][0].x);
			out.push_back(' ');
		}
		out.push_back('\n');
	}

	out += "\nmax2\n";
	for (int i = 0; i < 7; i++) {
		for (int j = 0; j < 7; j++) {
			out += to_string(maxL2[i][j][15].w);
			out.push_back(' ');
		}
		out.push_back('\n');
	}

	out += "\nmax2 index\n";
	for (int i = 0; i < 7; i++) {
		for (int j = 0; j < 7; j++) {
			out += to_string(imaxL2[i][j][0].x);
			out.push_back(' ');
		}
		out.push_back('\n');
	}

	out += "\nmax2 index\n";
	for (int i = 0; i < 7; i++) {
		for (int j = 0; j < 7; j++) {
			out += to_string(imaxL2[i][j][15].w);
			out.push_back(' ');
		}
		out.push_back('\n');
	}

	out += "\nconv3\n";
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			out += to_string(convL3[i][j][0].x);
			out.push_back(' ');
		}
		out.push_back('\n');
	}

	out += "\nconv3\n";
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			out += to_string(convL3[i][j][31].w);
			out.push_back(' ');
		}
		out.push_back('\n');
	}

	out += "\nmax3\n";
	out += to_string(maxL3[0].x);
	out.push_back(' ');
	out += to_string(maxL3[0].y);
	out.push_back('\n');
	out += to_string(maxL3[0].z);
	out.push_back(' ');
	out += to_string(maxL3[0].w);
	out.push_back('\n');

	out += "\nmax3\n";
	out += to_string(maxL3[127].x);
	out.push_back(' ');
	out += to_string(maxL3[127].y);
	out.push_back('\n');
	out += to_string(maxL3[127].z);
	out.push_back(' ');
	out += to_string(maxL3[127].w);
	out.push_back('\n');

	auto duration = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
	out += "\ntotal time: ";
	out += to_string(duration);
	out += "ms";
	cout << out << endl;

	system("pause");
	return 0;
}