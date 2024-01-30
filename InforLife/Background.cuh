#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <random>
#include "vector_types.h"
#include "lock.h"
using std::vector;
using std::rand;
constexpr int GENE_PLACE_NUM = 8;
constexpr int ACTION_PAIR_NUM = 3;
constexpr int DYNAMIC_LEVEL = 2;
float randf(float min, float max)
{
	float r = rand() % 10000 / 10000.0;
	return min + r * (max - min);
}

std::random_device rd;
std::mt19937 random_gen(rd());

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
	ActionPair conv_kernel_generater;
	ActionPair step;//��������˹�ͺ���
	ActionPair born;
	ActionPair death;
	float limit;
	DynamicData d_data;
	gene();
	~gene();
};

gene* DEFAULT_GENE;

struct divide_data
{
	float prob;
	float mutant_prob;
	float single_prob;
	float drift_mean;//û������Ļ�����0��
	float drift_std;
};

class Cell
{
private:
	int x;
	int y;
	Cells* group;//���Ըĳ��ں����ﴫ��
	gene* g;
public:
	Cell(int x, int y, Cells* e, gene* g);
	~Cell();
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
public:
	int size;
	int channel;
	int env_length;
	Cells(int size,int channel);
	//���ݻ�ȡ������
	gene** get_gene_mask() { return gene_mask; };
	bool* get_action_mask() { return action_mask; };
	float* get_dynamic() { return dynamic; };
	//����������
	void set_gene_belong(int x, int y, Cell* c);
	void generate_gene_belong(float r);
	void generate_g_mask(float r);//rӦ����gene_belong��һ����������genemask���ٵ���
	//�����̺�����
	void generate_gene_mask();
	void generate_action_mask();
	void generate_dynamic(float time);//�����߳�����
	void divide_cell();
	//��ѭ��������
	void step();
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
	Cells cells;
	vector<gene*>* gene_pool;//���Ҳ���
public:
	int const size;
	int const channel;
	Env(int size, int channel,Cells cells);
	gene* add_gene(gene g);
	void move_data(gene** gene_mask);
	void step();
};