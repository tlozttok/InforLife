#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
using std::vector;
class NonLinearMap
{
public:
	virtual float operator()(float x) = 0;
};

class ActionJudge
{
public:
	virtual bool operator()(float x) = 0;
};

struct gene
{
	int id;
	float* conv_kernel;
	int k_size;
	NonLinearMap* map;
	ActionJudge* born;
	ActionJudge* death;
};

class Env
{

};

class Cell
{
private:
	int x;
	int y;
	Env env;
	gene g;
public:
	Cell(int x, int y, Env e, gene g);
	~Cell();
	void mark_territory(float r);

};

class Env
{
private:
	float* data;
	vector<Cell>* gene_belong;
public:
	int size;
	void set_gene_belong(Cell* c);
	
};