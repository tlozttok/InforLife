#include "Background.cuh"
#include "DefaultPara.h"

void init_default_gene();

int main(int argc, char* argv[])
{
	Cells cell_group = Cells(ENV_SIZE, CHANNEL);
	Env env = Env(ENV_SIZE, CHANNEL, &cell_group);
	init_default_gene();
	Cell first = Cell(999, 999, &cell_group, DEFAULT_GENE);
	cell_group.add_cell(&first);
	cell_group.step(env.get_data_b(),env.get_data_d());
	env.step();
}

void init_default_gene() 
{
	cudaMallocManaged((void**)&DEFAULT_GENE, sizeof(gene));
	cudaMemcpy(DEFAULT_GENE->FCL_matrix, D_GENE_FCL_MATRIX, sizeof(float) * CHANNEL, cudaMemcpyHostToDevice);
	cudaMemcpy(DEFAULT_GENE->step.means, D_G_STEP_AP_MEAN, sizeof(float) * CHANNEL, cudaMemcpyHostToDevice);
	cudaMemcpy(DEFAULT_GENE->born.means, D_G_BORN_AP_MEAN, sizeof(float) * CHANNEL, cudaMemcpyHostToDevice);
	cudaMemcpy(DEFAULT_GENE->death.means, D_G_DEATH_AP_MEAN, sizeof(float) * CHANNEL, cudaMemcpyHostToDevice);
	cudaMemcpy(DEFAULT_GENE->step.stds, D_G_STEP_AP_STD, sizeof(float) * CHANNEL, cudaMemcpyHostToDevice);
	cudaMemcpy(DEFAULT_GENE->born.stds, D_G_BORN_AP_STD, sizeof(float) * CHANNEL, cudaMemcpyHostToDevice);
	cudaMemcpy(DEFAULT_GENE->death.stds, D_G_DEATH_AP_STD, sizeof(float) * CHANNEL, cudaMemcpyHostToDevice);
	cudaMemcpy(DEFAULT_GENE->d_data.A, D_G_DYNAMIC_A, sizeof(float) * CHANNEL, cudaMemcpyHostToDevice);
	cudaMemcpy(DEFAULT_GENE->d_data.phi, D_G_DYNAMIC_PHI, sizeof(float) * CHANNEL, cudaMemcpyHostToDevice);
	cudaMemcpy(DEFAULT_GENE->conv_kernel_generater[0].means, D_G_KERNEL_AP_MEAN, sizeof(float) * CHANNEL, cudaMemcpyHostToDevice);
	cudaMemcpy(DEFAULT_GENE->conv_kernel_generater[0].stds, D_G_KERNEL_AP_STD, sizeof(float) * CHANNEL, cudaMemcpyHostToDevice);
	cudaMemcpy(DEFAULT_GENE->conv_kernel_generater[1].means, D_G_KERNEL_AP_MEAN, sizeof(float) * CHANNEL, cudaMemcpyHostToDevice);
	cudaMemcpy(DEFAULT_GENE->conv_kernel_generater[1].stds, D_G_KERNEL_AP_STD, sizeof(float) * CHANNEL, cudaMemcpyHostToDevice);
	cudaMemcpy(DEFAULT_GENE->conv_kernel_generater[2].means, D_G_KERNEL_AP_MEAN, sizeof(float) * CHANNEL, cudaMemcpyHostToDevice);
	cudaMemcpy(DEFAULT_GENE->conv_kernel_generater[2].stds, D_G_KERNEL_AP_STD, sizeof(float) * CHANNEL, cudaMemcpyHostToDevice);
	cudaMemcpy(DEFAULT_GENE->weight, D_GENE_WEIGHT, sizeof(float) * CHANNEL, cudaMemcpyHostToDevice);
}
