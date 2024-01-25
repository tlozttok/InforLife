#include "Background.cuh"

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
		means[i] = randf(-1, 1);
		stds[i] = randf(0, 1);
	}
}

ActionPair::ActionPair(int pair_num)
{
	means = new float[ACTION_PAIR_NUM];
	stds = new float[ACTION_PAIR_NUM];
	for (int i = 0; i < pair_num; i++) {
		means[i] = randf(-1, 1);
		stds[i] = randf(0, 1);
	}
}

ActionPair::~ActionPair() { delete[] means; delete[] stds; }

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
						group->set_gene_belong(ix,iy,g);
					}
				}
			}
		}
	}
}