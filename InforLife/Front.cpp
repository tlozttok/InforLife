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
	memset((void*)gene_belong, 0, sizeof(gene_belong));//不可读数据不加锁
	for (auto cell : cell_group) {
		cell->mark_territory(r);
	}
}

void nearest_detect(Cell** gene_belong, int size, gene** result)
{
	float min_distance;
	Cell* nearest_cell;
	int i = 0;
	for (int ix = 0; ix < size; ix++) {
		for (int iy = 0; iy < size; iy++) {
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

void Cells::divide_cell()
{
	for (int i = 0; i < size * size; i++) {
		if (randf(0, 1) < d_data.prob) {

		}
	}
}

void Cells::step()
{
}
