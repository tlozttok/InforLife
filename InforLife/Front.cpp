#include "Background.cuh"
#include "DefaultPara.h"
#include <algorithm>
extern gene* DEFAULT_GENE;

void nearest_detect(Cell** gene_belong, int size, gene** result)
{
	float min_distance;
	Cell* nearest_cell;
	int i = 0;
	for (int iy = 0; iy < size; iy++) {
		for (int ix = 0; ix < size; ix++) {
			min_distance = INFINITY;
			nearest_cell = nullptr;
			for (int ic = 0; ic < GENE_PLACE_NUM; ic++) {
				Cell* cell = gene_belong[i * GENE_PLACE_NUM + ic];
				if (cell != nullptr) {
					float distance = (cell->X() - ix) * (cell->X() - ix) + (cell->Y() - iy) * (cell->Y() - iy);
					if (distance < min_distance) {
						min_distance = distance;
						nearest_cell = cell;
					}
				}
			}
			if (nearest_cell == nullptr) {
				result[i] = DEFAULT_GENE;
			}
			else {
				result[i] = nearest_cell->get_gene();
			}
			i++;
		}
	}
}

float mutant(float x, float single_prob, std::normal_distribution<>& d)
{
	if (randf(0, 1) < single_prob) {
		return x + d(random_gen);
	}
	else {
		return x;
	}
}

void array_mutant(const float* o_array, float* n_array, int length, float single_prob, std::normal_distribution<>& d)
{
	for (int i = 0; i < length; i++) {
		n_array[i] = mutant(o_array[i], single_prob, d);
	}
}

ActionPair ActionPair_mutant(const ActionPair a, float single_prob, std::normal_distribution<>& d)
{
	ActionPair na(a.num);
	array_mutant(a.means, na.means, a.num, single_prob, d);
	array_mutant(a.stds, na.stds, a.num, single_prob, d);
	return na;
}

DynamicData Dynamic_mutant(const DynamicData dd, float single_prob, std::normal_distribution<>& d)
{
	DynamicData nd;
	array_mutant(dd.A, nd.A, dd.level, single_prob, d);
	array_mutant(dd.phi, nd.phi, dd.level, single_prob, d);
	return nd;
}

void gene_mutant(const gene* parent, gene* n_gene, float single_prob, std::normal_distribution<>& d)
{
	array_mutant(parent->FCL_matrix, n_gene->FCL_matrix, parent->channel * parent->channel, single_prob, d);
	array_mutant(parent->weight, n_gene->weight, parent->channel, single_prob, d);
	n_gene->death = ActionPair_mutant(parent->death, single_prob, d);
	n_gene->born = ActionPair_mutant(parent->born, single_prob, d);
	n_gene->step = ActionPair_mutant(parent->step, single_prob, d);
	n_gene->d_data = Dynamic_mutant(parent->d_data, single_prob, d);
	n_gene->limit = mutant(parent->limit, single_prob, d);
	n_gene->id = rand();
	for (int c = 0; c < parent->channel; c++) {
		n_gene->conv_kernel_generater[c] = ActionPair_mutant(parent->conv_kernel_generater[c], single_prob, d);
	}
	n_gene->generate_kernels();
}

gene* gene_mutant_cuda(const gene* parent, divide_data d_data)//不要传cpu数据进去
{
	gene* n_gene = 0;
	cudaMallocManaged((void**)&n_gene, sizeof(gene));
	std::normal_distribution<> d(d_data.drift_mean, d_data.drift_std);
	gene_mutant(parent, n_gene, d_data.single_prob, d);
	return n_gene;
}

int pos2offset(int x, int y, int c, int size, int channel)
{
	int s = 0;
	int c_size = size * size;
	s += c_size * c;
	s += size * y;
	s += x;
	return s;
}

int3 offset2pos(int offset, int size, int channel)
{
	int c_size = size * size;
	int3 p = int3();
	p.x = offset % c_size % size;
	p.y = int((offset % c_size) / size);
	p.z = int(offset / c_size);
	return p;
}

void Cell::mark_territory(float r) 
{
	for (int ix = int(x - r - 1); ix<int(x + r + 1); ix++) {
		if (ix >= 0 && ix < group->size) {
			for (int iy = int(y - r - 1); iy<int(y + r + 1); iy++) {
				if (iy >= 0 && iy < group->size) {
					double d = sqrt((ix - x) * (ix - x) + (iy - y) * (iy - y));
					if (d < r) {
						group->set_gene_belong(ix,iy,this);
					}
				}
			}
		}
	}
}

void Cell::mark_territory(float r, Cell** gene_belong, int size)
{
	for (int ix = int(x - r - 1); ix<int(x + r + 1); ix++) {
		if (ix >= 0 && ix < group->size) {
			for (int iy = int(y - r - 1); iy<int(y + r + 1); iy++) {
				if (iy >= 0 && iy < group->size) {
					double d = sqrt((ix - x) * (ix - x) + (iy - y) * (iy - y));
					if (d < r) {
						int index = (iy * size + ix) * GENE_PLACE_NUM;
						for (int i = index; i < index + 9; i++) {
							if (gene_belong[i] == nullptr) {
								gene_belong[i] = this;
							}
						}
					}
				}
			}
		}
	}
}

float Cell::get_dynamic(float time)
{
	float d = 0.0f;
	for (int i = 0; i < g->d_data.level; i++) {
		d += g->d_data.A[i] * (sin(time * 2 * PI * i + g->d_data.phi[i]));
	}
	return d;
}

void Env::randomlise()
{
	float* r = new float[size * size * channel];
	for (int i = 0; i < size * size * channel; i++) {
		r[i] = randf(0, 1);
	}
	cudaMemcpy(data, r, sizeof(float) * size * size * channel, cudaMemcpyDefault);
}

void Cells::set_gene_belong(int x, int y, Cell* c)
{
	int index = (y * size + x) * GENE_PLACE_NUM;
	for (int i = index; i < index + 9; i++) {
		if (gene_belong[i] == nullptr) {
			gene_belong[i] = c;
		}
	}
}

void Cells::generate_gene_belong(float r)
{
	memset((void*)gene_belong, 0, sizeof(gene_belong));//不可读数据不加锁
	for (auto cell : cell_group) {
		cell->mark_territory(r);
	}
}

void Cells::generate_g_mask(float r)
{
	Cell** g_belong = new Cell * [env_length * GENE_PLACE_NUM]();
	gene** g_mask = new gene * [env_length];
	for (auto cell : cell_group) {
		cell->mark_territory(r-1,g_belong,size);
	}
	nearest_detect(g_belong, size, g_mask);
	delete[] g_belong;
	for (int i = 0; i < env_length; i++) {
		if ((gene_mask[i] == nullptr)!=(g_mask[i]==nullptr)) {
			g_mask[i] = gene_mask[i];
		}
		else g_mask[i] = nullptr;
	}
	delete[] generate_mask;
	generate_mask = g_mask;
}

void Cells::generate_gene_mask()
{
	gene** n_gene_mask = new gene*[env_length];
	nearest_detect(gene_belong, size, n_gene_mask);
	cell_territory_lock.lock();//写数据强制加锁
	delete[] gene_mask;
	gene_mask = n_gene_mask;
	cell_territory_lock.unlock();
}

void Cells::generate_action_mask()
{
	bool* n_action_mask = new bool[env_length]();
	for (int i = 0; i < env_length; i++) {
		if (generate_mask[i] != nullptr) {
			n_action_mask[i] = true;
		}
	}
	int i;
	for (auto cell : cell_group) {
		i = cell->X() + size * cell->Y();
		n_action_mask[i] = true;
	}
	cell_territory_lock.lock();//写数据强制加锁
	delete[] action_mask;
	action_mask = n_action_mask;
	cell_territory_lock.unlock();
}

void Cells::generate_dynamic(float time)
{
	float *n_dynamic = new float[env_length]();
	for (auto cell : cell_group) {
		int id = cell->Y() * size + cell->X();
		n_dynamic[id] = cell->get_dynamic(time);
	}
	dynamic_lock.lock();
	delete[] dynamic;
	dynamic = n_dynamic;
	dynamic_lock.unlock();
}

void Cells::divide_cell(float* data_b)
{
	for (int i = 0; i < size * size; i++) {
		if (generate_mask[i] != nullptr) {
			if (data_b[i]>0) {
				gene* new_gene;
				if (randf(0, 1) < d_data.mutant_prob) {
					new_gene = gene_mutant_cuda(generate_mask[i], d_data);
					reference_count[new_gene] = 0;
				}
				else {
					new_gene = generate_mask[i];
				}
				Cell* new_cell = new Cell(i % size, int(i / size + 1), this, new_gene);
				cell_group.push_back(new_cell);
				reference_count[new_gene]++;
				data_update_msg();
			}

		}
	}
}

void Cells::cell_die(float* data_d)
{
	for (auto cell = cell_group.begin(); cell != cell_group.end();) {
		int id = (*cell)->X() + size * (*cell)->Y();
		if (data_d[id] > 0) {
			reference_count[(*cell)->get_gene()]--;
			if (reference_count[(*cell)->get_gene()] <= 0) {
				reference_count.erase((*cell)->get_gene());
			}
			delete (*cell);
			std::iter_swap(cell, cell_group.end() - 1);
			cell_group.pop_back();
			data_update_msg();
		}
		else {
			cell++;
		}
	}
}

void Cells::step(float* g_data_b,float* g_data_d)
{
	if (gpu_data_lock.try_lock()) {
		float* data_b = new float[env_length];
		float* data_d = new float[env_length];
		cudaMemcpy(data_b, g_data_b, sizeof(float) * env_length, cudaMemcpyDeviceToHost);
		cudaMemcpy(data_d, g_data_d, sizeof(float) * env_length, cudaMemcpyDeviceToHost);
		gpu_data_lock.unlock();
		divide_cell(data_b);
		cell_die(data_d);
		delete[] data_b;
		delete[] data_d;
	}
	if (need_update) {
		generate_gene_belong(CELL_BELONG_RADIUS);
		generate_gene_mask();
		generate_g_mask(CELL_BELONG_RADIUS);
		generate_action_mask();
		need_update = false;
	}
}
