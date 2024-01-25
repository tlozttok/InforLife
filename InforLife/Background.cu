#include "Background.cuh"
#include "Function.cuh"
typedef float (*f2f)(float);

__device__ int pos2offset(int x, int y, int c, int size, int channel)
{
	int s = 0;
	int c_size = size * size;
	s += c_size * c;
	s += size * y;
	s += x;
	return s;
}

__device__ int3 offset2pos(int offset, int size, int channel)
{
	int c_size = size * size;
	int3 p = int3();
	p.x = offset % c_size % size;
	p.y = int((offset % c_size) / size);
	p.z = int(offset / c_size);
	return p;
}

__device__ void conv(float* data, float* kernel, int k_size, int channel, int x, int y, int d_size, float* result)
{
	int d_offset = pos2offset(x - k_size, y - k_size, 0, d_size, channel);
	int k_offset = 0;
	int dc_size = d_size * d_size;
	int kc_size = k_size * k_size;
	for (int c = 0; c < channel; c++) {
		float r = 0;
		for (int ix = x - k_size; ix <= x + k_size; ix++) {
			for (int iy = y - k_size; iy <= y + k_size; iy++) {
				r += data[d_offset] * kernel[k_offset];
				d_offset++;
				k_offset++;
			}
			d_offset += d_size;
			k_offset += k_size;
		}
		result[c] = r;
		d_offset += dc_size;
		k_offset += kc_size;
	}
}

__device__ void activate(float* data, int channel, f2f af) {
	for (int i = 0; i < channel; i++) {
		data[i] = af(data[i]);
	}
}

__device__ void matmul(float* data, float* mat, int width, int height, float* result) {
	int count = 0;
	int r;
	for (int iy = 0; iy < height; iy++) {
		r = 0;
		for (int ix = 0; ix < width; ix++) {
			r += mat[count] * data[ix];
			count++;
		}
		result[iy] = r;
	}
}

__device__ void respond(float* data, int channel, ActionPair RF) {
	for (int i = 0; i < channel; i++) {
		data[i] = mix_gaussian(data[i], RF);
	}
}


__global__ void step_compute(float* data, gene* gene, float* gene_mask, float* data_b, float* data_d, int size, int channel) 
{

}

Env::Env(int size, int channel):size(size)
{
	cudaStatus = cudaMallocManaged((void**)&data, 4 * size * size * channel);
	cudaStatus = cudaMallocManaged((void**)&data_b, 4 * size * size);
	cudaStatus = cudaMallocManaged((void**)&data_d, 4 * size * size);
	cudaStatus = cudaMallocManaged((void**)&gene_mask, 4 * size * size);
	
}

void Env::step()
{

}