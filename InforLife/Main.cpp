
#include "Background.cuh"
#include "DefaultPara.h"
#include <iostream>
#include <opencv2/core/utils/logger.hpp>
gene* DEFAULT_GENE;

using std::cout;
using std::endl;

void init_default_gene();

int main(int argc, char* argv[])
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
	cout << "start" << endl;
	Cells cell_group = Cells(ENV_SIZE, CHANNEL);
	cout << "cell_group init done" << endl;
	Env env = Env(ENV_SIZE, CHANNEL, &cell_group);
	cout << "env init done" << endl;
	init_default_gene();

	//cv::Mat kernel((DEFAULT_GENE->k_length * 2 + 1), (DEFAULT_GENE->k_length * 2 + 1), CV_8UC3);
	//trans_data(kernel, DEFAULT_GENE->conv_kernel, DEFAULT_GENE->channel, DEFAULT_GENE->k_length * 2 + 1);
	//cv::Mat large(700, 700, CV_8UC3);
	//cv::namedWindow("Kernel", cv::WINDOW_NORMAL);
	//cv::resizeWindow("Kernel", 700, 700);
	//cv::resize(kernel, large, cv::Size(700, 700), 0, 0, cv::INTER_NEAREST);
	//cv::imshow("Kernel", large);
	//cv::waitKey(0);

	cout << "default_gene init done" << endl;
	Cell first = Cell(45, 45, &cell_group, DEFAULT_GENE);
	cout << "first cell init done" << endl;
	//cell_group.add_cell(&first);
	cout << "add cell done" << endl;
	//env.randomlise();
	int i=1;
	while (i != 0) {
		cell_group.step(env.get_data_b(), env.get_data_d());
		cell_group.generate_dynamic(env.get_time());
		cout << "cell_group steped" << endl;
		env.step();
		cout << "env steped" << endl;
		cv::Mat image(ENV_SIZE, ENV_SIZE, CV_8UC3);
		env.get_data_img(image);
		cv::Mat large(700, 700, CV_8UC3);
		cv::namedWindow("Image", cv::WINDOW_NORMAL);
		cv::resizeWindow("Image", 700, 700);
		cv::resize(image, large, cv::Size(700, 700), 0, 0, cv::INTER_NEAREST);
		cv::imshow("Image", large);
		cv::waitKey(1);
	}
	return 0;
}

void init_default_gene() 
{
	cudaMallocManaged((void**)&DEFAULT_GENE, sizeof(gene));
	gene* n_gene = new gene();
	cudaMemcpy(DEFAULT_GENE, n_gene, sizeof(gene), cudaMemcpyDefault);
	cudaMemcpy(DEFAULT_GENE->FCL_matrix, D_GENE_FCL_MATRIX, sizeof(float) * CHANNEL * CHANNEL, cudaMemcpyDefault);
	cudaMemcpy(DEFAULT_GENE->step.means, D_G_STEP_AP_MEAN, sizeof(float) * STEP_ACTION_PAIR_NUM, cudaMemcpyDefault);
	cudaMemcpy(DEFAULT_GENE->born.means, D_G_BORN_AP_MEAN, sizeof(float) * ACTION_PAIR_NUM, cudaMemcpyDefault);
	cudaMemcpy(DEFAULT_GENE->death.means, D_G_DEATH_AP_MEAN, sizeof(float) * ACTION_PAIR_NUM, cudaMemcpyDefault);
	cudaMemcpy(DEFAULT_GENE->step.stds, D_G_STEP_AP_STD, sizeof(float) * STEP_ACTION_PAIR_NUM, cudaMemcpyDefault);
	cudaMemcpy(DEFAULT_GENE->born.stds, D_G_BORN_AP_STD, sizeof(float) * ACTION_PAIR_NUM, cudaMemcpyDefault);
	cudaMemcpy(DEFAULT_GENE->death.stds, D_G_DEATH_AP_STD, sizeof(float) * ACTION_PAIR_NUM, cudaMemcpyDefault);
	cudaMemcpy(DEFAULT_GENE->d_data.A, D_G_DYNAMIC_A, sizeof(float) * DYNAMIC_LEVEL, cudaMemcpyDefault);
	cudaMemcpy(DEFAULT_GENE->d_data.phi, D_G_DYNAMIC_PHI, sizeof(float) * DYNAMIC_LEVEL, cudaMemcpyDefault);
	for (int c = 0; c < CHANNEL; c++) {
		cudaMemcpy(DEFAULT_GENE->conv_kernel_generater[c].means, D_G_KERNEL_AP_MEAN, sizeof(float) * KERNEL_PAIR_NUM, cudaMemcpyDefault);
		cudaMemcpy(DEFAULT_GENE->conv_kernel_generater[c].stds, D_G_KERNEL_AP_STD, sizeof(float) * KERNEL_PAIR_NUM, cudaMemcpyDefault);
	}
	cudaMemcpy(DEFAULT_GENE->weight, D_GENE_WEIGHT, sizeof(float) * CHANNEL, cudaMemcpyDefault);
	DEFAULT_GENE->generate_kernels();
	operator delete(n_gene);
}
