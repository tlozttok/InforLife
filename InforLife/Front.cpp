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

void ActionPair_mutant(const ActionPair* a, ActionPair* na,float single_prob, std::normal_distribution<>& d)
{
	array_mutant(a->means, na->means, a->num, single_prob, d);
	array_mutant(a->stds, na->stds, a->num, single_prob, d);
}

void Dynamic_mutant(const DynamicData* dd, DynamicData* nd, float single_prob, std::normal_distribution<>& d)
{
	array_mutant(dd->A, nd->A, dd->level, single_prob, d);
	array_mutant(dd->phi, nd->phi, dd->level, single_prob, d);
}

void gene_mutant(const gene* parent, gene* n_gene, float single_prob, std::normal_distribution<>& d)
{
	array_mutant(parent->FCL_matrix, n_gene->FCL_matrix, parent->channel * parent->channel, single_prob, d);
	array_mutant(parent->weight, n_gene->weight, parent->channel, single_prob, d);
	ActionPair_mutant(&parent->death, &(n_gene->death), single_prob, d);
	ActionPair_mutant(&parent->born, &(n_gene->born), single_prob, d);
	ActionPair_mutant(&parent->step, &(n_gene->step),single_prob, d);
	Dynamic_mutant(&parent->d_data, &(n_gene->d_data), single_prob, d);
	n_gene->limit = mutant(parent->limit, single_prob, d);
	n_gene->base = abs(mutant(parent->base, single_prob, d));
	n_gene->base_k = abs(mutant(parent->base, single_prob, d));
	n_gene->id = rand();
	for (int c = 0; c < parent->channel; c++) {
		ActionPair_mutant(&parent->conv_kernel_generater[c], &(n_gene->conv_kernel_generater[c]), single_prob, d);
	}
	n_gene->generate_kernels();
}

gene* gene_mutant_cuda(const gene* parent, divide_data d_data)//不要传cpu数据进去
{
	gene* c_n_gene = new gene();
	gene* n_gene = 0;
	cudaMallocManaged((void**)&n_gene, sizeof(gene));
	cudaMemcpy(n_gene, c_n_gene, sizeof(gene), cudaMemcpyDefault);
	operator delete(c_n_gene);
	std::normal_distribution<> d(d_data.drift_mean, d_data.drift_std);
	gene_mutant(parent, n_gene, d_data.single_prob, d);
	recorder.write("ng", sizeof(char) * 2);
	n_gene->Serialize(recorder);
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
	for (int ix = int(x - r - 1); ix<=int(x + r + 1); ix++) {
		if (ix >= 0 && ix < group->size) {
			for (int iy = int(y - r - 1); iy<=int(y + r + 1); iy++) {
				if (iy >= 0 && iy < group->size) {
					double d = sqrt((ix - x) * (ix - x) + (iy - y) * (iy - y));
					if (d < r) {
						int index = (iy * size + ix) * GENE_PLACE_NUM;
						for (int i = index; i < index + 9; i++) {
							if (gene_belong[i] == nullptr) {
								gene_belong[i] = this;
								break;
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

gene* Cell::get_new_gene()
{
	return gene_mutant_cuda(g,DEFAULT_D_DATA);
}

void Cell::Serialize(ofstream& file)
{
	file.write(reinterpret_cast<char*>(&id), sizeof(long));
	file.write(reinterpret_cast<char*>(&x), sizeof(int));
	file.write(reinterpret_cast<char*>(&y), sizeof(int));
	file.write(reinterpret_cast<char*>(&(g->id)), sizeof(int));
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
	for (int i = index; i < index + GENE_PLACE_NUM; i++) {
		if (gene_belong[i] == nullptr) {
			gene_belong[i] = c;
		}
	}
}

cv::Scalar id2color(int id)
{
	// 使用位操作和模运算来映射x到a, b, c
	int a = (id & 0x1F);
	int b = ((id >> 5) & 0x1F);
	int c = ((id >> 10) & 0x1F);

	a = (a * 31 + 52998) % 256;
	b = (b * 10 + 46451) % 256;
	c = (c * 54 + 41728) % 256;

	return cv::Scalar(a, b, c);
}

void Cells::draw_cells(cv::Mat mat ,int mat_size)
{
	for (auto cell : cell_group) {
		cv::circle(mat, cv::Point(cell->X()/float(size)* mat_size, cell->Y()/float(size)* mat_size), 2, id2color(cell->get_gene()->id), -1);
	}
}

void Cells::generate_gene_belong(float r)
{
	memset((void*)gene_belong, 0, sizeof(Cell**) * size * size * GENE_PLACE_NUM);//不可读数据不加锁
	for (auto cell : cell_group) {
		cell->mark_territory(r);
	}
}

void Cells::generate_g_mask(float r)
{
	cv::Mat kernel(size, size, CV_8UC3);
	//show_data(kernel, 1, size, gene_mask);
	Cell** g_belong = new Cell * [env_length * GENE_PLACE_NUM]();
	gene** g_mask = new gene * [env_length];
	for (auto cell : cell_group) {
		cell->mark_territory(r-1,g_belong,size);
	}
	nearest_detect(g_belong, size, g_mask);
	delete[] g_belong;
	//show_data(kernel, 1, size, g_mask);
	for (int i = 0; i < env_length; i++) {
		if ((gene_mask[i] == DEFAULT_GENE)!=(g_mask[i]==DEFAULT_GENE)) {
			g_mask[i] = gene_mask[i];
		}
		else g_mask[i] = DEFAULT_GENE;
	}
	//show_data(kernel, 1, size, g_mask);
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
	//printf("\t\tgenerate_action_mask_half\n");
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
	//printf("\tthere?\n");
	delete[] dynamic;
	dynamic = n_dynamic;
	dynamic_lock.unlock();
}

void Cells::divide_cell(float* data_b)
{
	for (int i = 0; i < size * size; i++) {
		if (generate_mask[i] != DEFAULT_GENE) {
			if (data_b[i]>0) {
				gene* new_gene;
				float v = randf(0, 1);
				if (v < d_data.mutant_prob) {
					new_gene = gene_mutant_cuda(generate_mask[i], d_data);
					reference_count[new_gene] = 0;
				}
				else {
					new_gene = generate_mask[i];
				}
				Cell* new_cell = new Cell(i % size, int(i / size), this, new_gene);
				last_id++;
				new_cell->set_id(last_id);
				cell_group.push_back(new_cell);
				reference_count[new_gene]++;
				recorder.write("cb", sizeof(char) * 2);
				new_cell->Serialize(recorder);
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
				if ((*cell)->get_gene() != DEFAULT_GENE) {
					gene* g = (*cell)->get_gene();
					recorder.write("gd", sizeof(char) * 2);
					recorder.write(reinterpret_cast<char*>(g->id), sizeof(int));
					g->~gene();
					cudaFree(g);
				}
			}
			recorder.write("cd", sizeof(char) * 2);
			recorder.write(reinterpret_cast<char*>((*cell)->id), sizeof(int));
			delete (*cell);
			data_update_msg();
			cell = cell_group.erase(cell);
		}
		else {
			cell++;
		}
	}
}

void Cells::step(float* g_data_b,float* g_data_d)
{
	if (gpu_data_lock.try_lock()) {
		//printf("\tget_gpu_data begin\n");
		float* data_b = new float[env_length];
		float* data_d = new float[env_length];
		cudaMemcpy(data_b, g_data_b, sizeof(float) * env_length, cudaMemcpyDeviceToHost);
		cudaMemcpy(data_d, g_data_d, sizeof(float) * env_length, cudaMemcpyDeviceToHost);
		gpu_data_lock.unlock();
		//printf("\tget_gpu_data\n");
		divide_cell(data_b);
		cell_die(data_d);
		//printf("\tprocess_gpu_data\n");
		delete[] data_b;
		delete[] data_d;
		//printf("\tdel_data\n");
	}
	if (need_update) {//有问题
		//printf("\tgenerate_gene_belong\n");
		generate_gene_belong(CELL_BELONG_RADIUS);
		//printf("\tgenerate_gene_mask\n");
		generate_gene_mask();
		//printf("\tgenerate_g_mask\n");
		generate_g_mask(CELL_BELONG_RADIUS);
		//printf("\tgenerate_action_mask\n");
		generate_action_mask();
		//printf("\tupdate_done\n");
		need_update = false;
	}
}
