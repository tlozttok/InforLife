#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <random>
#include "vector_types.h"
using std::vector;
using std::rand;
const int GENE_PLACE_NUM = 8;
const int ACTION_PAIR_NUM = 3;

float randf(float min, float max)
{
	float r = rand() % 10000 / 10000.0;
	return min + r * (max - min);
}



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

struct ActionPair
{
	int num;
	float* means;
	float* stds;
	ActionPair();
	ActionPair(int pair_num);
	~ActionPair();
};

struct gene
{
	int id;
	float* conv_kernel;
	int k_length;//(内核大小-1)/2
	int channel;
	float* FCL_matrix;
	float* weight;
	ActionPair step;//用两个高斯就好了
	ActionPair born;
	ActionPair death;
	float limit;
};

class Cells
{
private:
	gene** gene_belong;
	Cell* cell_group;
	int cell_num;
public:
	int size;
	void set_gene_belong(int x, int y, gene* g);
};

class Cell
{
private:
	int x;
	int y;
	Cells* group;
	gene* g;
public:
	Cell(int x, int y, Cells* e, gene* g);
	~Cell();
	void mark_territory(float r);

};

class Env
{
private:
	cudaError_t cudaStatus;
	float* data;//顺序c*h*w
	float* data_b;
	float* data_d;
	gene** gene_mask;
	bool* action_mask;
	vector<gene*>* gene_pool;//暂且不用
public:
	int const size;
	Env(int size, int channel);
	gene* add_gene(gene g);
	void move_data(gene** gene_mask);
	void step();
};