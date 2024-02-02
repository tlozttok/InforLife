#include "Background.cuh"
#include "DefaultPara.h"


float gaussian(float x, float mean, float std) {
	float coeff = 1.0f / (sqrtf(2.0f * PI ) * std);
	float exponent = expf(-0.5f * powf((x - mean) / std, 2));
	return coeff * exponent;
}

float randf(float min, float max)
{
	float r = rand() % 10000 / 10000.0;
	return min + r * (max - min);
}

float mix_gaussian(float x, ActionPair RF) {
	float r = 0;
	for (int i = 0; i < RF.num; i++) {
		r += gaussian(x, RF.means[i], RF.stds[i]);
	}
	return r;
}

ActionPair::ActionPair()
{
	cudaMallocManaged((void**)&means, sizeof(float) * ACTION_PAIR_NUM);
	cudaMallocManaged((void**)&stds, sizeof(float) * ACTION_PAIR_NUM);
	num = ACTION_PAIR_NUM;
	for (int i = 0; i < ACTION_PAIR_NUM; i++) {
		means[i] = randf(0, 1);
		stds[i] = randf(0, 0.1);
	}
}

ActionPair::ActionPair(int pair_num)
{
	cudaMallocManaged((void**)&means, sizeof(float) * pair_num);
	cudaMallocManaged((void**)&stds, sizeof(float) * pair_num);
	num = pair_num;
	for (int i = 0; i < pair_num; i++) {
		means[i] = randf(0, 1);
		stds[i] = randf(0, 0.1);
	}
}

ActionPair::~ActionPair() { cudaFree(means); cudaFree(stds); }

DynamicData::DynamicData()
{
	level = DYNAMIC_LEVEL;
	cudaMallocManaged((void**)&A, sizeof(float) * DYNAMIC_LEVEL);
	cudaMallocManaged((void**)&phi, sizeof(float) * DYNAMIC_LEVEL);
	for (int i = 0; i < DYNAMIC_LEVEL; i++) {
		A[i] = randf(0, 1);
		phi[i] = randf(0, 2 * PI);
	}
}

DynamicData::~DynamicData()
{
	cudaFree(A);
	cudaFree(phi);
}

void generate_kernel(int k_length, float* kernel, ActionPair* src, int channel, float* kernel_sum)
{
	int id = 0;
	for (int c = 0; c < channel; c++) {
		float sum = 0;
		for (int iy = -k_length; iy < k_length; iy++) {
			for (int ix = -k_length; ix < k_length; ix++) {
				float x = float(ix) / k_length;
				float y = float(iy) / k_length;
				float w = mix_gaussian(sqrt(x * x + y * y), src[c]);
				sum += w;
				kernel[id] = w;
				id++;
			}
		}
		kernel_sum[c] = sum;
	}
}

void gene::generate_kernels()
{
	generate_kernel(k_length, conv_kernel, conv_kernel_generater, channel, kernel_sum);//sum
}

gene::gene() {
	id = 0;
	k_length = KERNEL_LENGTH;
	int l = (k_length * 2 + 1) * (k_length * 2 + 1);
	channel = CHANNEL;
	cudaMallocManaged((void**)&conv_kernel, sizeof(float) * l);
	cudaMallocManaged((void**)&kernel_sum, sizeof(float) * channel);
	cudaMallocManaged((void**)&weight, sizeof(float) * channel);
	cudaMallocManaged((void**)&FCL_matrix, sizeof(float) * channel * channel);
	cudaMallocManaged((void**)&conv_kernel_generater, sizeof(ActionPair) * channel);
	step = ActionPair(STEP_ACTION_PARI_NUM);
	born = ActionPair();
	death = ActionPair();
	d_data = DynamicData();
	limit = 0;
}

gene::~gene()
{
	cudaFree(conv_kernel);
	cudaFree(kernel_sum);
	cudaFree(FCL_matrix);
	cudaFree(weight);
	for (int c = 0; c < channel; c++) {
		conv_kernel_generater[c].~ActionPair();
	}
	cudaFree(conv_kernel_generater);
}

Cell::Cell(int x, int y, Cells* e, gene* g) :x(x), y(y), group(e), g(g)
{

}

Cells::Cells(int size, int channel) : size(size), channel(channel) {
	env_length = size * size;
	gene_belong = new Cell * [env_length * GENE_PLACE_NUM]();
	gene_mask = new gene * [env_length]();
	generate_mask = new gene * [env_length]();
	action_mask = new bool[env_length]();
	dynamic = new float[env_length * channel]();
	cell_group = vector<Cell*>();
	cell_group.reserve(32);
	need_update = true;
};

Cells::~Cells()
{
	delete[] gene_belong;
	delete[] gene_mask;
	delete[] generate_mask;
	delete[] action_mask;
	delete[] dynamic;
}

