
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

const float case1[4][4] =
{
	0.1, 0.2, 0.3, 0.4,
	0.5, 0.6, 0.7, 0.8,
	0.9, 0.10, 0.11, 0.12,
	0.13, 0.14, 0.15, 0.16
};

const float case1Expect[3] =
{
	0.0, 0.0, 1.0
};

inline int clamp(int x, int y, int z)
{
	return min(max(x, y), z);
}

inline float fnRELU(float x) {
	return max(float(0.0), x);
}

inline float fnSig(float x) {
	return exp(x);
}

inline float fnELU(float x, float alpha) {
	return x >= 0. ? x : alpha * (exp(x) - 1.0);
}

int main()
{
	random_device rd;  //Will be used to obtain a seed for the random number engine
	mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	uniform_real_distribution<float> dis(-1.0, 1.0);

	// Learning rate
	float lr = 0.2;

	// Bias learning rate
	float lrb = 0.1;
	float input[4][4] = { 0.0 };

	// Filters
	float w[2][2][2] = { dis(gen), dis(gen),
		dis(gen), dis(gen),
		dis(gen), dis(gen),
		dis(gen), dis(gen) };

	// 2x2 Convolution
	float conv1[3][3][2] = { 0.0 };

	// Activation Func
	float actFn[3][3][2] = { 0.0 };

	// 2x2 Max Pool
	float maxP1[2][2][2] = { 0.0 };

	// 2x2 Max Pool source indicies
	int maxPSrc1[2][2][2] = { 0 };

	// Fully connected weights
	float fcw[8][3];
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 3; j++)
			fcw[i][j] = dis(gen);

	// Full connected net output
	float nout[3] = { 0.0 };

	// Softmax output
	float smOut[3] = { 0.0 };

	// Layer 1 bias deltas
	float b1[2] = { dis(gen), dis(gen) };

	// Layer 2 bias deltas
	float b2[3];
	for (int i = 0; i < 3; i++) b2[i] = dis(gen);

	// Full connected weight deltas
	float dfcw[8][3];
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 3; j++)
			dfcw[i][j] = 0.0;

	// Full connected input deltas
	float dmp1[8];
	for (int i = 0; i < 8; i++)
		dmp1[i] = 0.0;

	// Layer 1 bias deltas
	float db1[2] = { 0.0 };

	// Layer 2 bias deltas
	float db2[3] = { 0.0 };

	// Activation Func deltas
	float dactFn[3][3][2] = { 0.0 };

	// Filter gradient
	float dw[2][2][2] = { 0.0 };

	memcpy(input, case1, sizeof(case1));

	for (int ll = 0; ll < 10; ll++) {
		// 2x2 w
		for (int f = 0; f < 2; f++) {
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					float s = input[i][j] * w[0][0][f];
					s += input[i][j + 1] * w[0][1][f];
					s += input[i + 1][j] * w[1][0][f];
					s += input[i + 1][j + 1] * w[1][1][f];
					conv1[i][j][f] = s + b1[f];
					// Activation Func
					actFn[i][j][f] = fnELU(conv1[i][j][f], 0.2);
				}
			}
		}

		// Max pool
		for (int f = 0; f < 2; f++) {
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 2; j++) {
					float rmax = actFn[i][j][f];
					rmax = max(rmax, actFn[i][j + 1][f]);
					rmax = max(rmax, actFn[i + 1][j][f]);
					rmax = max(rmax, actFn[i + 1][j + 1][f]);
					maxP1[i][j][f] = rmax;
				}
			}
		}

		// Max pool index
		for (int f = 0; f < 2; f++) {
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 2; j++) {
					int i1 = i + 1;
					int j1 = j + 1;

					float rmax = actFn[i][j][f];
					maxPSrc1[i][j][f] = i * 3 + j;

					rmax = max(rmax, actFn[i][j1][f]);
					maxPSrc1[i][j][f] = (rmax == actFn[i][j1][f]) ?
						i * 3 + j1 : maxPSrc1[i][j][f];

					rmax = max(rmax, actFn[i1][j][f]);
					maxPSrc1[i][j][f] = (rmax == actFn[i1][j][f]) ?
						i1 * 3 + j : maxPSrc1[i][j][f];

					rmax = max(rmax, actFn[i1][j1][f]);
					maxPSrc1[i][j][f] = (rmax == actFn[i1][j1][f]) ?
						i1 * 3 + j1 : maxPSrc1[i][j][f];
				}
			}
		}

		// Fully connected layer
		for (int i = 0; i < 3; i++) {
			float s = 0.0;
			for (int j = 0; j < 8; j++) {
				int x = j / 2;
				x = x > 1 ? x - 2 : x;
				s += (maxP1[x][j % 2][j / 4] * fcw[j][i]);
			}
			nout[i] = s + b2[i];
		}

		// Softmax
		for (int i = 0; i < 3; i++) {
			smOut[i] = exp(nout[i]) / (exp(nout[0]) + exp(nout[1]) + exp(nout[2]));
		}

		// Backprop

		// Fully Connected layer weight gradient
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 8; j++) {
				int x = j / 2;
				x = x > 1 ? x - 2 : x;
				dfcw[j][i] = (smOut[i] - case1Expect[i]) * (smOut[i] * (1.0 - smOut[i])) *
					maxP1[x][j % 2][j / 4];
			}
		}

		// Bias 2 gradient
		for (int i = 0; i < 3; i++) {
			db2[i] = (smOut[i] - case1Expect[i]);
		}

		// Fully Connected layer input gradient
		for (int j = 0; j < 8; j++) {
			dmp1[j] = 0.;
			for (int i = 0; i < 3; i++) {
				dmp1[j] += (smOut[i] - case1Expect[i]) * (smOut[i] * (1.0 - smOut[i])) *
					fcw[j][i];
			}
		}

		// Restructure input gradient
		for (int f = 0; f < 2; f++) {
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {

					// Max pool source index
					int ic = clamp(i, 0, 2);
					int jc = clamp(j, 0, 2);
					int ilo = clamp(i - 1, 0, 2);
					int jlo = clamp(j - 1, 0, 2);
					int c = (i * 3 + j);

					// Check all possible places it could be
					bool b1 = maxPSrc1[ic][jc][f] == c;
					bool b2 = maxPSrc1[ic][jlo][f] == c;
					bool b3 = maxPSrc1[ilo][jc][f] == c;
					bool b4 = maxPSrc1[ilo][jlo][f] == c;

					// Find the right index, 2x2x2 -> 8x1 
					int x = b1 ? jc : b2 ? jlo : b3 ? jc : b4 ? jlo : -1;
					int y = b1 ? ic : b2 ? ic : b3 ? ilo : b4 ? ilo : -1;
					int di = (y > -1) && (x > -1) ? y * 2 + x + f * 4 : -1;

					dactFn[i][j][f] = b1 && di > -1 ? dmp1[di] : dactFn[i][j][f];
					dactFn[i][j][f] = b2 && di > -1 ? dmp1[di] : dactFn[i][j][f];
					dactFn[i][j][f] = b3 && di > -1 ? dmp1[di] : dactFn[i][j][f];
					dactFn[i][j][f] = b4 && di > -1 ? dmp1[di] : dactFn[i][j][f];
				}
			}
		}

		// Filter gradients
		for (int f = 0; f < 2; f++) {
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 2; j++) {
					int i1 = i + 1;
					int i2 = i + 2;
					int j1 = j + 1;
					int j2 = j + 2;
					dw[i][j][f] =
						input[i][j] * dactFn[0][0][f] + input[i][j1] * dactFn[0][1][f] +
						input[i][j2] * dactFn[0][2][f] + input[i1][j] * dactFn[1][0][f] +
						input[i1][j1] * dactFn[1][1][f] + input[i1][j2] * dactFn[1][2][f] +
						input[i2][j] * dactFn[2][0][f] + input[i2][j1] * dactFn[2][1][f] +
						input[i2][j2] * dactFn[2][2][f];
				}
			}
		}

		// Bias 1 gradient
		for (int f = 0; f < 2; f++) {
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					db1[f] += dactFn[i][j][f];
				}
			}
		}

		// New weights based on gradient
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 8; j++) {
				fcw[j][i] -= lr * dfcw[j][i];
			}
		}

		// New filters
		for (int f = 0; f < 2; f++) {
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 2; j++) {
					w[i][j][f] -= lr * dw[i][j][f];
				}
			}
		}

		// Bias update
		for (int i = 0; i < 2; i++) {
			b1[i] -= lrb * db1[i];
		}

		for (int i = 0; i < 3; i++) {
			b2[i] -= lrb * db2[i];
		}

		//for (int i = 0; i < 3; i++) {
		//	for (int j = 0; j < 3; j++) {
		//		clog << conv1[i][j][0] << " ";
		//	}
		//	clog << endl;
		//}
		//clog << endl;

		//for (int i = 0; i < 3; i++) {
		//	for (int j = 0; j < 3; j++) {
		//		clog << conv1[i][j][1] << " ";
		//	}
		//	clog << endl;
		//}
		//clog << endl;

		//for (int i = 0; i < 2; i++) {
		//	for (int j = 0; j < 2; j++) {
		//		clog << maxP1[i][j][0] << " ";
		//	}
		//	clog << endl;
		//}
		//clog << endl;

		//for (int i = 0; i < 2; i++) {
		//	for (int j = 0; j < 2; j++) {
		//		clog << maxP1[i][j][1] << " ";
		//	}
		//	clog << endl;
		//}
		//clog << endl;

		//for (int i = 0; i < 2; i++) {
		//	for (int j = 0; j < 2; j++) {
		//		clog << maxPSrc1[i][j][0] << " ";
		//	}
		//	clog << endl;
		//}
		//clog << endl;

		//for (int i = 0; i < 2; i++) {
		//	for (int j = 0; j < 2; j++) {
		//		clog << maxPSrc1[i][j][1] << " ";
		//	}
		//	clog << endl;
		//}
		//clog << endl;

		//for (int i = 0; i < 3; i++) {
		//	clog << nout[i] << " ";
		//}
		//clog << endl;

		for (int i = 0; i < 3; i++) {
			clog << smOut[i] << " ";
		}
		clog << endl << endl;

		//for (int i = 0; i < 3; i++) {
		//	for (int j = 0; j < 8; j++) {
		//		clog << dfcw[j][i] << " ";
		//	}
		//	clog << endl;
		//}
		//clog << endl;

		//for (int j = 0; j < 8; j++) {
		//	clog << dmp1[j] << " ";
		//}
		//clog << endl << endl;

		//for (int i = 0; i < 3; i++) {
		//	for (int j = 0; j < 3; j++) {
		//		clog << dactFn[i][j][0] << " ";
		//	}
		//	clog << endl;
		//}
		//clog << endl;

		//for (int i = 0; i < 3; i++) {
		//	for (int j = 0; j < 3; j++) {
		//		clog << dactFn[i][j][1] << " ";
		//	}
		//	clog << endl;
		//}
		//clog << endl;

		//for (int i = 0; i < 2; i++) {
		//	for (int j = 0; j < 2; j++) {
		//		clog << dw[i][j][0] << " ";
		//	}
		//	clog << endl;
		//}
		//clog << endl;

		//for (int i = 0; i < 2; i++) {
		//	for (int j = 0; j < 2; j++) {
		//		clog << dw[i][j][1] << " ";
		//	}
		//	clog << endl;
		//}
		//clog << endl;

		//for (int i = 0; i < 2; i++) {
		//	clog << b1[i] << " ";
		//}
		//clog << endl;

		//for (int i = 0; i < 3; i++) {
		//	clog << b2[i] << " ";
		//}
		//clog << endl << endl;

		//system("pause");
	}
	system("pause");
	return 0;
}