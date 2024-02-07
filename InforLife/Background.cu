#include "Background.cuh"
#include "Function.cuh"
#include <cstdio>
#include "DefaultPara.h"

extern gene* DEFAULT_GENE;
typedef float (*f2f)(float);

__device__ int pos2offset(int x, int y, int c, int size)
{
	int s = 0;
	int c_size = size * size;
	s += c_size * c;
	s += size * y;
	s += x;
	return s;
}

__device__ int3 offset2pos(int offset, int size)
{
	int c_size = size * size;
	int3 p = int3();
	p.x = offset % c_size % size;
	p.y = int((offset % c_size) / size);
	p.z = int(offset / c_size);
	return p;
}

__device__ void conv(float* data, float* kernel, float* sum, int k_size, int channel, int x, int y, int d_size, float* result)//需重做
{
	int d_offset;
	int k_offset = 0;
	float r;
	for (int c = 0; c < channel; c++) {
		r = 0;
		for (int iy = y - k_size; iy <= y + k_size; iy++) {
			d_offset = pos2offset(x - k_size, iy, c, d_size);
			for (int ix = x - k_size; ix <= x + k_size; ix++) {
				if (ix >= 0 && ix < d_size && iy >= 0 && iy < d_size) {
					r += data[d_offset] * kernel[k_offset];
				}
				d_offset++;
				k_offset++;
			}
		}
		float s = sum[c];
		result[c] = r / sum[c];
		//printf("\tc_r:%f", result[c]);
	}
}

__device__ void activate(float* data, int channel, f2f af) {
	for (int i = 0; i < channel; i++) {
		data[i] = af(data[i]);
	}
}

__device__ void matmul(float* data, float* mat, int width, int height, float* result) {
	int count = 0;
	float r;
	for (int iy = 0; iy < height; iy++) {
		r = 0;
		for (int ix = 0; ix < width; ix++) {
			r += mat[count] * data[ix];
			count++;
		}
		result[iy] = r;
	}
}

__device__ void respond(float* data, int channel, ActionPair* RF) {
	for (int i = 0; i < channel; i++) {
		data[i] = mix_gaussian_gpu(data[i], RF);
	}
}


__global__ void step_compute(
	float* data,
	gene** gene_mask, 
	float* dynamic, 
	float delta_t,
	bool* action_mask, 
	float* data_b, 
	float* data_d, 
	float* n_data,
	int size, 
	int channel,
	float* buffer_conv_r,
	float* buffer_mat_r
) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int num = size * size;
	int amount = int(num / blockDim.x + 1);
	for (int i = id * amount; i < (id+1) * amount; i++) {
		if (i < num) {
			//printf("\ni:%d", i);
			int3 pos = offset2pos(i, size);
			gene* g = gene_mask[i];

			conv(data, g->conv_kernel, g->kernel_sum, g->k_length, channel, pos.x, pos.y, size, buffer_conv_r+channel*i);//卷积
			//printf("\tr_data:%f\t%f\t%f", (buffer_conv_r + channel * i)[0], (buffer_conv_r + channel * i)[1], (buffer_conv_r + channel * i)[2]);
			activate(buffer_conv_r + channel * i, channel, sin_af);//激活
			//printf("\tr_data:%f\t%f\t%f", (buffer_conv_r + channel * i)[0], (buffer_conv_r + channel * i)[1], (buffer_conv_r + channel * i)[2]);
			matmul(buffer_conv_r + channel * i, g->FCL_matrix, channel, channel, buffer_mat_r+channel * i);//矩阵乘法
			//printf("\tr_data:%f\t%f\t%f", (buffer_mat_r + channel * i)[0], (buffer_mat_r + channel * i)[1], (buffer_mat_r + channel * i)[2]);
			respond(buffer_mat_r + channel * i, channel, &(g->step));//得到结果
			//printf("\tr_data:%f\t%f\t%f", (buffer_mat_r + channel * i)[0], (buffer_mat_r + channel * i)[1], (buffer_mat_r + channel * i)[2]);
			//if (dynamic[i] != 0.0f) {
			//	printf("\tid:%d,d:%f", i, dynamic[i]);
			//}
			for (int c = 0; c < channel; c++) {
				//n_data[i + c * num] = sigmoid_g(data[i + c * num]+(buffer_mat_r + channel * i)[c] * delta_t + dynamic[i]);
				//n_data[i + c * num] = __saturatef(data[i + c * num] + (buffer_mat_r + channel * i)[c] * delta_t + dynamic[i]);
				n_data[i + c * num] = data[i + c * num] + ((buffer_mat_r + channel * i)[c] - (g->base)) * delta_t + dynamic[i];
			}

			if (action_mask[i]) {
				float born = 0;
				float death = 0;

				matmul(buffer_conv_r + channel * i, g->weight, channel, 1, &born);//细胞动作计算

				death = born;

				respond(&born, 1, &(g->born));

				respond(&death, 1, &(g->death));

				death -= g->limit;
				born -= g->limit;
				data_b[i] = born;
				data_d[i] = death;

			}
		}
	}
}

Env::Env(int size, int channel, Cells* cells):size(size),channel(channel),cells(cells)
{
	cudaStatus = cudaMallocManaged((void**)&data, sizeof(float) * size * size * channel);
	cudaStatus = cudaMallocManaged((void**)&data_b, sizeof(float) * size * size);
	cudaStatus = cudaMallocManaged((void**)&data_d, sizeof(float) * size * size);
	cudaStatus = cudaMallocManaged((void**)&gene_mask, sizeof(gene*) * size * size);
	cudaStatus = cudaMallocManaged((void**)&dynamic, sizeof(float) * size * size);//多通道
	cudaStatus = cudaMallocManaged((void**)&action_mask, sizeof(bool) * size * size);
	cudaMemset(data, 0, sizeof(float) * size * size * channel);
	cudaMemset(data_b, 0, sizeof(float) * size * size);
	cudaMemset(data_d, 0, sizeof(float) * size * size);
	cudaMemset(gene_mask, 0, sizeof(gene*) * size * size);
	cudaMemset(dynamic, 0, sizeof(float) * size * size);
	cudaMemset(action_mask, 0, sizeof(bool) * size * size);
	time = 0;
	delta_t = DELTA_T;
}

Env::~Env()
{
	cudaStatus = cudaFree(data);
	cudaStatus = cudaFree(data_b);
	cudaStatus = cudaFree(data_d);
	cudaStatus = cudaFree(gene_mask);
	cudaStatus = cudaFree(dynamic);
	cudaStatus = cudaFree(action_mask);
}

void Env::step()
{
	if (cell_territory_lock.try_lock())//尝试读取数据
	{
		gene** genemask = cells->get_gene_mask();
		bool* actionmask = cells->get_action_mask();
		cudaMemcpy(gene_mask, genemask, sizeof(gene*) * size * size, cudaMemcpyDefault);
		cudaMemcpy(action_mask, actionmask, sizeof(bool) * size * size, cudaMemcpyDefault);
		cell_territory_lock.unlock();
	}
	if (dynamic_lock.try_lock())//尝试读取数据
	{
		float* d = cells->get_dynamic();
		cudaMemcpy(dynamic, d, sizeof(float) * size * size, cudaMemcpyDefault);
		dynamic_lock.unlock();
	}
	float* ndata = 0;//旧数据保留以供其他线程读取
	float* n_data_b = 0;
	float* n_data_d = 0;
	cudaStatus = cudaMallocManaged((void**)&ndata, sizeof(float) * size * size * channel);
	cudaStatus = cudaMallocManaged((void**)&n_data_b, sizeof(float) * size * size);
	cudaStatus = cudaMallocManaged((void**)&n_data_d, sizeof(float) * size * size);
	float* buffer_c = 0;
	float* buffer_m = 0;
	cudaStatus = cudaMalloc((void**)&buffer_c, sizeof(float) * channel * size * size);
	cudaStatus = cudaMalloc((void**)&buffer_m, sizeof(float) * channel * size * size);
	step_compute<<<32,64>>>(data, gene_mask, dynamic, delta_t, action_mask, n_data_b, n_data_d, ndata, size, channel, buffer_c, buffer_m);
	checkCuda(cudaGetLastError());
	time += delta_t;
	cudaDeviceSynchronize();
	cudaFree(buffer_c);
	cudaFree(buffer_m);
	gpu_data_lock.lock();//坚持覆写数据
	cudaFree(data);
	cudaFree(data_b);
	cudaFree(data_d);
	data = ndata;
	data_b = n_data_b;
	data_d = n_data_d;
	gpu_data_lock.unlock();
}

void Env::get_data_img(cv::Mat mat)
{
	uchar* data_ptr = mat.data;
	int s0 = mat.step[0];
	int s1 = mat.step[1];

	int i = 0;
	for (int c = 0; c < channel; c++) {
		int id = 0;
		for (int col = 0; col < size; col++) {
			for (int r = 0; r < size; r++) {
				int t = int(data[i] * 255) %255;
				*(data_ptr + col * s0 + r * s1 + c) = t;
				i++;
			}
		}
	}
}