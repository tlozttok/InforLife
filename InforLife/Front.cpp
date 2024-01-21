#include "Background.cuh"

Cell::Cell(int x, int y, Env e, gene g) :x(x), y(y), env(e), g(g) 
{

}

void Cell::mark_territory(float r) 
{
	for (int ix = int(x - r - 1); ix<int(x + r + 1); ix++) {
		if (ix >= 0 && ix < env.size) {
			for (int iy = int(y - r - 1); iy<int(y + r + 1); iy++) {
				if (iy >= 0 && iy < env.size) {
					double d = sqrt((ix - x) * (ix - x) + (iy - y) * (iy - y));
					if (d < r) {
						env.set_gene_belong(this);
					}
				}
			}
		}
	}
}