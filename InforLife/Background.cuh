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
#include <cstdio>
#include <opencv2/opencv.hpp>

#ifndef checkCuda
#define checkCuda(err)  __checkCudaErrors (err, __FILE__, __LINE__)

using std::ofstream;

inline void __checkCudaErrors(cudaError_t err, const char* file, const int line)
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
			err, cudaGetErrorString(err), file, line);//getCudaDrvErrorString
		exit(EXIT_FAILURE);
	}
}
#endif

typedef std::vector<std::vector<std::vector<uint8_t>>> vector_img;
using std::map;
using std::vector;
using std::rand;
float randf(float min, float max);

static std::random_device rd;
static std::mt19937 random_gen(100);


struct ActionPair
{
	int num;
	float* means;
	float* stds;
	ActionPair();
	ActionPair(int pair_num);
	~ActionPair();
	void Serialize(ofstream& file);
};

struct DynamicData
{
	float* A;
	//弧度制
	float* phi;
	int level;
	DynamicData();
	~DynamicData();
	void Serialize(ofstream& file);
};

struct gene
{
	int id;
	float* conv_kernel;//顺序[c,h,w]
	int k_length;//(内核大小-1)/2
	float* kernel_sum;
	int channel;
	float* FCL_matrix;//顺序[h,w]
	float* weight;
	ActionPair* conv_kernel_generater;
	ActionPair step;//用两个高斯就好了
	ActionPair born;
	ActionPair death;
	float limit;
	float base;
	float base_k;
	DynamicData d_data;
	gene();
	void generate_kernels();
	~gene();
	void Serialize(ofstream& file);
};


struct divide_data
{
	float prob;
	float mutant_prob;
	float single_prob;
	float drift_mean;//没有意外的话就是0了
	float drift_std;
};

constexpr divide_data DEFAULT_D_DATA = { 0.1,0.05,0.4,0.0,0.005 };

class Cells;

class Cell
{
private:
	int x;
	int y;
	Cells* group;//可以改成在函数里传递
	gene* g;
public:
	int id;
	Cell(int x, int y, Cells* e, gene* g);
	void set_id(int cell_id) { id = cell_id; };
	void mark_territory(float r);
	void mark_territory(float r, Cell** gene_belong, int size);
	float get_dynamic(float time);
	int X() { return x; };
	int Y() { return y; };
	gene* get_gene() { return g; };
	gene* get_new_gene();
	~Cell() {};
	void Serialize(ofstream& file);
};

class Cells
{
private:
	long last_id = 0;
	Cell** gene_belong;//顺序[h,w,place_num]
	gene** gene_mask;//可被其他线程读
	gene** generate_mask;
	bool need_update;
	bool* action_mask;//可被其他线程读
	float* dynamic;//可被其他线程读
	vector<Cell*> cell_group;
	divide_data d_data;
	map<gene*, int> reference_count;
	//辅助函数：
	
	void generate_gene_belong(float r);
	//r应当和gene_belong的一样，在生成genemask后再调用
	void generate_g_mask(float r);
	void data_update_msg() { need_update = true; };
	//主流程函数：

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
	void add_cell(Cell* c) { cell_group.push_back(c); reference_count[c->get_gene()]++; };
	void draw_cells(cv::Mat mat, int mat_size);
	//数据获取函数：

	gene** get_gene_mask() { return gene_mask; };
	bool* get_action_mask() { return action_mask; };
	float* get_dynamic() { return dynamic; };
	//辅助函数：

	void set_gene_belong(int x, int y, Cell* c);
	//主流程函数：

	//单独线程运行
	void generate_dynamic(float time);
	//主循环函数：
	
	//传GPU数据进去
	void step(float* data_b,float* data_d);
};

class Env
{
private:
	cudaError_t cudaStatus;
	float* data;//顺序[c,h,w]
	float* data_b;
	float* data_d;
	gene** gene_mask;
	bool* action_mask;
	float* dynamic;//顺序[c,h,w]，和data一样
	float delta_t;
	float time;
	Cells* cells;
	vector<gene*>* gene_pool;//暂且不用
public:
	int const size;
	int const channel;
	Env(int size, int channel,Cells* cells);
	~Env();
	void step();
	void randomlise();
	void get_data_img(cv::Mat mat);
	float* get_data_b() { return data_b; };
	float* get_data_d() { return data_d; };
	float get_time() { return time; };
};

float gaussian(float x, float mean, float std);
void trans_data(cv::Mat mat, float* data, int channel, int size);
inline void show_data(cv::Mat mat, int channel, int size, gene** data)
{
		uchar* data_ptr = mat.data;
		int s0 = mat.step[0];
		int s1 = mat.step[1];

		int i = 0;
		for (int c = 0; c < channel; c++) {
			int id = 0;
			for (int col = 0; col < size; col++) {
				for (int r = 0; r < size; r++) {
					int t = data[i]->id==0?0:255;
					*(data_ptr + col * s0 + r * s1 + c) = t;
					i++;
				}
			}
		}

		cv::Mat large(700, 700, CV_8UC3);
		cv::namedWindow("Kernel", cv::WINDOW_NORMAL);
		cv::resizeWindow("Kernel", 700, 700);
		cv::resize(mat, large, cv::Size(700, 700), 0, 0, cv::INTER_NEAREST);
		cv::imshow("Kernel", large);
		cv::waitKey(0);
}
#endif // !BACKGROUND