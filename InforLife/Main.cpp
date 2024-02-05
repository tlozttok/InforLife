
#include "Background.cuh"
#include "DefaultPara.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <opencv2/core/utils/logger.hpp>
gene* DEFAULT_GENE;

using std::cout;
using std::endl;

void init_default_gene();
void read_config(const std::string& filename);

int main(int argc, char* argv[])
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
	read_config("E:\\code\\c++\\InforLife\\config.txt");
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
	Cell* first = new Cell(45, 45, &cell_group, DEFAULT_GENE);
	cout << "first cell init done" << endl;
	cell_group.add_cell(first);
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

void set_config(std::string key, std::istringstream& iss) {
	if (key=="APN"){
		iss >> ACTION_PAIR_NUM;
		D_G_BORN_AP_MEAN = new float[ACTION_PAIR_NUM];
		D_G_DEATH_AP_MEAN = new float[ACTION_PAIR_NUM];
		D_G_BORN_AP_STD = new float[ACTION_PAIR_NUM];
		D_G_DEATH_AP_STD = new float[ACTION_PAIR_NUM];
	}
	else if (key == "SAPN") {
		iss >> STEP_ACTION_PAIR_NUM;
		D_G_STEP_AP_MEAN = new float[STEP_ACTION_PAIR_NUM];
		D_G_STEP_AP_STD = new float[STEP_ACTION_PAIR_NUM];
	}
	else if (key == "KPN") {
		iss >> KERNEL_PAIR_NUM;
		D_G_KERNEL_AP_MEAN = new float[KERNEL_PAIR_NUM];
		D_G_KERNEL_AP_STD = new float[KERNEL_PAIR_NUM];
	}
	else if (key == "DL") {
		iss >> DYNAMIC_LEVEL;
		D_G_DYNAMIC_A = new float[DYNAMIC_LEVEL];
		D_G_DYNAMIC_PHI = new float[DYNAMIC_LEVEL];
	}
	else if (key == "CBR") {
		iss >> CELL_BELONG_RADIUS;
	}
	else if (key == "ES") {
		iss >> ENV_SIZE;
	}
	else if (key == "KL") {
		iss >> KERNEL_LENGTH;
	}
	else if (key == "C") {
		iss >> CHANNEL;
		D_GENE_WEIGHT = new float[CHANNEL];
		D_GENE_FCL_MATRIX = new float[CHANNEL * CHANNEL];
	}
	else if (key == "DT") {
		iss >> DELTA_T;
	}
	else if (key == "DGW") {
		for (int i = 0; i < CHANNEL; i++) {
			iss >> D_GENE_WEIGHT[i];
		}
	}
	else if (key == "DGFM") {
		for (int i = 0; i < CHANNEL*CHANNEL; i++) {
			iss >> D_GENE_FCL_MATRIX[i];
		}
	}
	else if (key == "DGSAM") {
		for (int i = 0; i < STEP_ACTION_PAIR_NUM; i++) {
			iss >> D_G_STEP_AP_MEAN[i];
		}
	}
	else if (key == "DGBAM") {
		for (int i = 0; i < ACTION_PAIR_NUM; i++) {
			iss >> D_G_BORN_AP_MEAN[i];
		}
	}
	else if (key == "DGDAM") {
		for (int i = 0; i < ACTION_PAIR_NUM; i++) {
			iss >> D_G_DEATH_AP_MEAN[i];
		}
	}
	else if (key == "DGSAS") {
		for (int i = 0; i < STEP_ACTION_PAIR_NUM; i++) {
			iss >> D_G_STEP_AP_STD[i];
		}
	}
	else if (key == "DGBAS") {
		for (int i = 0; i < ACTION_PAIR_NUM; i++) {
			iss >> D_G_BORN_AP_STD[i];
		}
	}
	else if (key == "DGDAS") {
		for (int i = 0; i < ACTION_PAIR_NUM; i++) {
			iss >> D_G_DEATH_AP_STD[i];
		}
	}
	else if (key == "DGDA") {
		for (int i = 0; i < DYNAMIC_LEVEL; i++) {
			iss >> D_G_DYNAMIC_A[i];
		}
	}
	else if (key == "DGDP") {
		for (int i = 0; i < DYNAMIC_LEVEL; i++) {
			iss >> D_G_DYNAMIC_PHI[i];
		}
	}
	else if (key == "DGKAM") {
		for (int i = 0; i < KERNEL_PAIR_NUM; i++) {
			iss >> D_G_KERNEL_AP_MEAN[i];
		}
	}
	else if (key == "DGKAS") {
		for (int i = 0; i < KERNEL_PAIR_NUM; i++) {
			iss >> D_G_KERNEL_AP_STD[i];
		}
	}
}

void read_config(const std::string& filename)
{
	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "无法打开配置文件： " << filename << std::endl;
		return;
	}
	std::string line;
	while (std::getline(file, line)) {
		if (line.empty() || line[0] == '#') {
			continue;
		}
		std::istringstream iss(line);
		std::string key;
		iss >> key;
		set_config(key, iss);
	}
}
