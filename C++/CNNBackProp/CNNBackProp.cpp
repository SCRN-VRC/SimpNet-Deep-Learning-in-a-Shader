
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

class CNN {
private:
	float L1[7][7][6]; // kern = 3, stride = 2
	float L2[3][3][12]; // kern = 3, stride = 1
	float FC1[24];
	float FC2[6];
	float wL1[3][3][6][12];
	float wL2[3][3][12][24];
	float wFC1[24][6];

public:
	CNN()
	{
		L1 = { 0.0f };
		L2 = { 0.0f };
		FC1 = { 0.0f };
		FC2 = { 0.0f };

		random_device rd;  //Will be used to obtain a seed for the random number engine
		mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
		normal_distribution<float> dis1(0.0f, 1.0f / sqrt(648.0f));

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 6; k++) {
					for (int l = 0; l < 12; l++) {
						wL1[i][j][k][l] = dis1(gen);
					}
				}
			}
		}
	}
};

int main()
{
	system("pause");
	return 0;
}