#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <string.h>
#include <time.h>
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double diff_sigmoid(double x) { return (sigmoid(x)*(1.0 - sigmoid(x))); }

int main() {
	const char *number[20] = { "0.txt","1.txt","2.txt","3.txt","4.txt","5.txt","6.txt","7.txt","8.txt","9.txt",
		"error_0.txt","error_1.txt","error_2.txt","error_3.txt","error_4.txt","error_5.txt","error_6.txt","error_7.txt","error_8.txt","error_9.txt", };
	FILE *fp;
	int input[226];
	double hidden[41];
	double output[10];
	double v[226][41];
	double w[41][10];
	double d_v[226][41];
	double d_w[41][10];
	double z_in[41];
	double y_in[10];
	double d_in[41];
	double alpha = 0.35;
	double sum = 0;
	double d_k[10];
	double t[10];
	double d_j[41];
	int i, j, k, count = 0, x = 0;
	int check = 0;
	srand((unsigned)time(NULL));

	//step 0
	for (i = 0; i <= 225; i++) //input -> hidden 가중치 초기화 (-0.5,0.5)
		for (j = 0; j <= 40; j++)
			v[i][j] = (rand() % 2) - 0.5;

	for (i = 0; i <= 40; i++) //hidden -> output 가중치 초기화(-0.5, 0.5)
		for (j = 0; j < 10; j++)
			w[i][j] = (rand() % 2) - 0.5;

	for (int i = 0; i < 10; i++)
		output[i] = 0;

	input[0] = 1; hidden[0] = 1.0; //bias 초기화

	for (int x = 0; x < 5; x++)//시행횟수
		for (int num = 0; num < 10; num++) {
			fp = fopen(number[num], "r");
			for (i = 1; i <= 225; i++) //Input layer 초기화
				fscanf(fp, "%d", &input[i]);
			fclose(fp);

			for (i = 0; i < 10; i++) //t_k 초기화
				t[i] = 0.0;
			t[num] = 1.0;

			while (output[num] <= 0.97) {//step7 & 학습에대한 한계치(적중도 0.9 이상시 패스)
				printf("\n%d회차\n", count++);
				//step 1,2
				for (j = 1; j <= 40; j++) {
					z_in[j] = v[0][j];
					for (i = 1; i <= 225; i++)
						z_in[j] += (input[i] * v[i][j]);
					hidden[j] = sigmoid(z_in[j]);
				}
				//step 3
				for (k = 0; k < 10; k++) {
					y_in[k] = w[0][k];
					for (j = 1; j <= 40; j++)
						y_in[k] += (hidden[j] * w[j][k]);
					output[k] = sigmoid(y_in[k]);
				}
				//step 4
				for (k = 0; k < 10; k++) {
					d_k[k] = (t[k] - output[k]) * diff_sigmoid(y_in[k]);
					d_w[0][k] = alpha * d_k[k];
					for (j = 1; j <= 40; j++)
						d_w[j][k] = alpha * d_k[k] * hidden[j];
				}
				//step 5
				for (j = 1; j <= 40; j++) {
					d_in[j] = 0;
					for (k = 0; k < 10; k++)
						d_in[j] += (d_k[k] * w[j][k]);
					d_j[j] = d_in[j] * diff_sigmoid(z_in[j]);
				}
				for (j = 1; j <= 40; j++)
					for (i = 0; i <= 225; i++)
						d_v[i][j] = alpha * d_j[j] * input[i];
				//step 6
				for (j = 1; j <= 40; j++)
					for (k = 0; k < 10; k++)
						w[j][k] += d_w[j][k];
				for (i = 1; i <= 225; i++)
					for (j = 1; j <= 40; j++)
						v[i][j] += d_v[i][j];
			}
		}
	for (int num = 10; num < 20; num++) { //의도적 오류 삽입 숫자에 대한 구별 확인
		double max = 0;
		int predic = 0;
		fp = fopen(number[num], "r");
		for (i = 1; i <= 225; i++) 
			fscanf(fp, "%d", &input[i]);
		fclose(fp);
		for (j = 1; j <= 40; j++) {
			z_in[j] = v[0][j]; 
			for (i = 1; i <= 225; i++)
				z_in[j] += (input[i] * v[i][j]);
			hidden[j] = sigmoid(z_in[j]);
		}
		for (k = 0; k < 10; k++) {
			y_in[k] = w[0][k]; 
			for (j = 1; j <= 40; j++)
				y_in[k] += (hidden[j] * w[j][k]);
			output[k] = sigmoid(y_in[k]);
			if (output[k] > max) { // 가장 output값이 높은 숫자 확인
				max = output[k];
				predic = k;
			}
		}
		for (i = 1; i <= 225; i++) {
			if (input[i] == 1)
				printf("■");
			else
				printf("□");
			if (i % 15 == 0) printf("\n");
		}
		printf("인식한 숫자 : %d", predic);
		printf("\n\n");
	}

}