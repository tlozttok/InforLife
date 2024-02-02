#pragma once
#ifndef BACKGROUND
#define BACKGROUND

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <random>
#include <map>
#include "vector_types.h"
#include "lock.h"
using std::map;
using std::vector;
using std::rand;
float randf(float min, float max);

static std::random_device rd;
static std::mt19937 random_gen(rd());

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

struct DynamicData
{
	float* A;
	//������
	float* phi;
	int level;
	DynamicData();
	~DynamicData();
};

struct gene
{
	int id;
	float* conv_kernel;//˳��[c,h,w]
	int k_length;//(�ں˴�С-1)/2
	float* kernel_sum;
	int channel;
	float* FCL_matrix;//˳��[h,w]
	float* weight;
	ActionPair* conv_kernel_generater;
	ActionPair step;//��������˹�ͺ���
	ActionPair born;
	ActionPair death;
	float limit;
	DynamicData d_data;
	gene();
	void generate_kernels();
	~gene();
};

static gene* DEFAULT_GENE;

struct divide_data
{
	float prob;
	float mutant_prob;
	float single_prob;
	float drift_mean;//û������Ļ�����0��
	float drift_std;
};

constexpr divide_data DEFAULT_D_DATA = { 0.1,0.2,0.4,0.0,0.01 };

class Cells;

class Cell
{
private:
	int x;
	int y;
	Cells* group;//���Ըĳ��ں����ﴫ��
	gene* g;
public:
	Cell(int x, int y, Cells* e, gene* g);
	void mark_territory(float r);
	void mark_territory(float r, Cell** gene_belong, int size);
	float get_dynamic(float time);
	int X() { return x; };
	int Y() { return y; };
	gene* get_gene() { return g; };
};

class Cells
{
private:
	Cell** gene_belong;//˳��[h,w,place_num]
	gene** gene_mask;//�ɱ������̶߳�
	gene** generate_mask;
	bool need_update;
	bool* action_mask;//�ɱ������̶߳�
	float* dynamic;//�ɱ������̶߳�
	vector<Cell*> cell_group;
	divide_data d_data;
	map<gene*, int> reference_count;
	//����������

	void generate_gene_belong(float r);
	//rӦ����gene_belong��һ����������genemask���ٵ���
	void generate_g_mask(float r);
	void data_update_msg() { need_update = true; };
	//�����̺�����

	void divide_cell(float* data_b);
	void cell_die(float* data_d);
	void generate_gene_mask();
	void generate_action_mask();

public:
	int size;
	int channel;
	int env_length;
	Cells(int size,int channel);
	~Cells();
	void add_cell(Cell* c) { cell_group.push_back(c); };
	//���ݻ�ȡ������

	gene** get_gene_mask() { return gene_mask; };
	bool* get_action_mask() { return action_mask; };
	float* get_dynamic() { return dynamic; };
	//����������

	void set_gene_belong(int x, int y, Cell* c);
	//�����̺�����

	//�����߳�����
	void generate_dynamic(float time);
	//��ѭ��������
	
	//��GPU���ݽ�ȥ
	void step(float* data_b,float* data_d);
};

class Env
{
private:
	cudaError_t cudaStatus;
	float* data;//˳��[c,h,w]
	float* data_b;
	float* data_d;
	gene** gene_mask;
	bool* action_mask;
	float* dynamic;//˳��[c,h,w]����dataһ��
	float delta_t;
	float time;
	Cells* cells;
	vector<gene*>* gene_pool;//���Ҳ���
public:
	int const size;
	int const channel;
	Env(int size, int channel,Cells* cells);
	~Env();
	void step();
	float* get_data_b() { return data_b; };
	float* get_data_d() { return data_d; };
};

float gaussian(float x, float mean, float std);

#endif // !BACKGROUND