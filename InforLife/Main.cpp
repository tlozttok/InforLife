
#include "Background.cuh"
#include "DefaultPara.h"
#include <iostream>
#include <opencv2/opencv.hpp>


using std::cout;
using std::endl;

void init_default_gene();

int main(int argc, char* argv[])
{
	cout << "start" << endl;
	Cells cell_group = Cells(ENV_SIZE, CHANNEL);
	cout << "cell_group init done" << endl;
	Env env = Env(ENV_SIZE, CHANNEL, &cell_group);
	cout << "env init done" << endl;
	init_default_gene();
	checkCuda(cudaGetLastError());
	cout << "default_gene init done" << endl;
	Cell first = Cell(45, 45, &cell_group, DEFAULT_GENE);
	checkCuda(cudaGetLastError());
	cout << "first cell init done" << endl;
	cell_group.add_cell(&first);
	checkCuda(cudaGetLastError());
	cout << "add cell done" << endl;
	int i=1;
	while (i != 0) {
		cell_group.step(env.get_data_b(), env.get_data_d());
		cout << "cell_group steped" << endl;
		env.step();
		cout << "env steped" << endl;
		int* data = env.get_data_img();
		cv::Mat image(ENV_SIZE, ENV_SIZE, CV_8UC3, data);
		cv::imshow("Image", image);
		delete[] data;
		cv::waitKey(0);
	}
	return 0;
}

void init_default_gene() 
{
	cudaMallocManaged((void**)&DEFAULT_GENE, sizeof(gene));
	gene n_gene = gene();
	checkCuda(cudaGetLastError());
	cudaMemcpy(DEFAULT_GENE, &n_gene, sizeof(gene), cudaMemcpyDefault);
	checkCuda(cudaGetLastError());
	cudaMemcpy(DEFAULT_GENE->FCL_matrix, D_GENE_FCL_MATRIX, sizeof(float) * CHANNEL * CHANNEL, cudaMemcpyDefault);
	checkCuda(cudaGetLastError());
	float* m = DEFAULT_GENE->step.means;
	m[0] = 0.5;
	cudaMemcpy(m, D_G_STEP_AP_MEAN, sizeof(float) * STEP_ACTION_PARI_NUM, cudaMemcpyDefault);
	checkCuda(cudaGetLastError());
	cudaMemcpy(DEFAULT_GENE->born.means, D_G_BORN_AP_MEAN, sizeof(float) * ACTION_PAIR_NUM, cudaMemcpyDefault);
	checkCuda(cudaGetLastError());
	cudaMemcpy(DEFAULT_GENE->death.means, D_G_DEATH_AP_MEAN, sizeof(float) * ACTION_PAIR_NUM, cudaMemcpyDefault);
	checkCuda(cudaGetLastError());
	cudaMemcpy(DEFAULT_GENE->step.stds, D_G_STEP_AP_STD, sizeof(float) * ACTION_PAIR_NUM, cudaMemcpyDefault);
	checkCuda(cudaGetLastError());
	cudaMemcpy(DEFAULT_GENE->born.stds, D_G_BORN_AP_STD, sizeof(float) * ACTION_PAIR_NUM, cudaMemcpyDefault);
	checkCuda(cudaGetLastError());
	cudaMemcpy(DEFAULT_GENE->death.stds, D_G_DEATH_AP_STD, sizeof(float) * ACTION_PAIR_NUM, cudaMemcpyDefault);
	checkCuda(cudaGetLastError());
	cudaMemcpy(DEFAULT_GENE->d_data.A, D_G_DYNAMIC_A, sizeof(float) * DYNAMIC_LEVEL, cudaMemcpyDefault);
	checkCuda(cudaGetLastError());
	cudaMemcpy(DEFAULT_GENE->d_data.phi, D_G_DYNAMIC_PHI, sizeof(float) * DYNAMIC_LEVEL, cudaMemcpyDefault);
	checkCuda(cudaGetLastError());
	cudaMemcpy(DEFAULT_GENE->conv_kernel_generater[0].means, D_G_KERNEL_AP_MEAN, sizeof(float) * KERNEL_PAIR_NUM, cudaMemcpyDefault);
	checkCuda(cudaGetLastError());
	cudaMemcpy(DEFAULT_GENE->conv_kernel_generater[0].stds, D_G_KERNEL_AP_STD, sizeof(float) * KERNEL_PAIR_NUM, cudaMemcpyDefault);
	checkCuda(cudaGetLastError());
	cudaMemcpy(DEFAULT_GENE->conv_kernel_generater[1].means, D_G_KERNEL_AP_MEAN, sizeof(float) * KERNEL_PAIR_NUM, cudaMemcpyDefault);
	checkCuda(cudaGetLastError());
	cudaMemcpy(DEFAULT_GENE->conv_kernel_generater[1].stds, D_G_KERNEL_AP_STD, sizeof(float) * KERNEL_PAIR_NUM, cudaMemcpyDefault);
	checkCuda(cudaGetLastError());
	cudaMemcpy(DEFAULT_GENE->conv_kernel_generater[2].means, D_G_KERNEL_AP_MEAN, sizeof(float) * KERNEL_PAIR_NUM, cudaMemcpyDefault);
	checkCuda(cudaGetLastError());
	cudaMemcpy(DEFAULT_GENE->conv_kernel_generater[2].stds, D_G_KERNEL_AP_STD, sizeof(float) * KERNEL_PAIR_NUM, cudaMemcpyDefault);
	checkCuda(cudaGetLastError());
	cudaMemcpy(DEFAULT_GENE->weight, D_GENE_WEIGHT, sizeof(float) * CHANNEL, cudaMemcpyDefault);
	checkCuda(cudaGetLastError());
}
