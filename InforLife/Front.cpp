#include "Background.cuh"
constexpr auto PI = 3.14159265358979323846;


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

ActionPair::ActionPair()
{
	means = new float[ACTION_PAIR_NUM];
	stds = new float[ACTION_PAIR_NUM];
	for (int i = 0; i < ACTION_PAIR_NUM; i++) {
		means[i] = randf(0, 1);
		stds[i] = randf(0, 0.1);
	}
}

ActionPair::ActionPair(int pair_num)
{
	means = new float[pair_num];
	stds = new float[pair_num];
	for (int i = 0; i < pair_num; i++) {
		means[i] = randf(0, 1);
		stds[i] = randf(0, 0.1);
	}
}

ActionPair::~ActionPair() { delete[] means; delete[] stds; }

DynamicData::DynamicData()
{
	level = DYNAMIC_LEVEL;
	A = new float[DYNAMIC_LEVEL];
	phi = new float[DYNAMIC_LEVEL];
	for (int i = 0; i < DYNAMIC_LEVEL; i++) {
		A[i] = randf(0, 1);
		phi[i] = randf(0, 2*PI);
	}
}

DynamicData::~DynamicData()
{
	delete[] A;
	delete[] phi;
}

Cell::Cell(int x, int y, Cells* e, gene* g) :x(x), y(y), group(e), g(g) 
{
	
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
		d += g->d_data.A[i] * (sin(time * 2 * PI + g->d_data.phi[i]));
	}
	return d;
}

Cells::Cells(int size,int channel) : size(size),channel(channel) { 
	env_length = size * size;
	gene_belong = new Cell* [env_length * GENE_PLACE_NUM]();
	gene_mask = new gene* [env_length]();
	generate_mask= new gene* [env_length]();
	action_mask = new bool[env_length]();
	dynamic = new float[env_length * channel]();
	cell_group = vector<Cell*>();
	cell_group.reserve(32);
	need_update = true;
};

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
	memset((void*)gene_belong, 0, sizeof(gene_belong));//���ɶ����ݲ�����
	for (auto cell : cell_group) {
		cell->mark_territory(r);
	}
}

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

void Cells::generate_g_mask(float r)
{
	Cell** g_belong = new Cell * [env_length * GENE_PLACE_NUM]();
	gene** g_mask = new gene * [env_length];
	for (auto cell : cell_group) {
		cell->mark_territory(r-1,g_belong,size);
	}
	nearest_detect(g_belong, size, g_mask);
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
	gene** n_gene_mask = new gene * [env_length];
	nearest_detect(gene_belong, size, n_gene_mask);
	cell_territory_lock.lock();//д����ǿ�Ƽ���
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
	cell_territory_lock.lock();//д����ǿ�Ƽ���
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

float gaussian(float x, float mean, float std) {
	float coeff = 1.0f / (sqrtf(2.0f * PI) * std);//Ϊ��ƽ��
	float exponent = expf(-0.5f * powf((x - mean) / std, 2));
	return coeff * exponent;
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

ActionPair ActionPair_mutant(const ActionPair a,float single_prob, std::normal_distribution<>& d)
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
	n_gene->death = ActionPair_mutant(parent->death, single_prob,d);
	n_gene->born = ActionPair_mutant(parent->born, single_prob,d);
	n_gene->step = ActionPair_mutant(parent->step, single_prob,d);
	n_gene->d_data = Dynamic_mutant(parent->d_data, single_prob,d);
}

gene* gene_mutant_cuda(const gene* parent,divide_data d_data)//��Ҫ��cpu���ݽ�ȥ
{
	gene* cpu_p_gene = new gene();
	gene* cpu_n_gene = new gene();
	gene* n_gene = 0;
	std::normal_distribution<> d(d_data.drift_mean, d_data.drift_std);
	cudaMemcpy(cpu_p_gene, parent, sizeof(gene), cudaMemcpyDeviceToHost);
	gene_mutant(cpu_p_gene, cpu_n_gene, d_data.single_prob, d);
	cudaMalloc((void**)&n_gene, sizeof(gene));
	cudaMemcpy(n_gene, cpu_n_gene, sizeof(gene), cudaMemcpyHostToDevice);
	delete cpu_p_gene;
	delete cpu_n_gene;
}

void Cells::divide_cell()
{
	for (int i = 0; i < size * size; i++) {
		if (generate_mask[i] != nullptr) {
			if (randf(0, 1) < d_data.prob) {
				gene* new_gene;
				if (randf(0, 1) < d_data.mutant_prob) {
					new_gene = gene_mutant_cuda(generate_mask[i], d_data);
				}
				else {
					new_gene = generate_mask[i];
				}
				Cell* new_cell = new Cell(i % size, int(i / size + 1), this, new_gene);
				cell_group.push_back(new_cell);
			}

		}
	}
}

void Cells::step()
{
}
