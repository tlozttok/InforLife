#include "Background.cuh"
#include <corecrt_math_defines.h>

__device__ inline float gaussian_gpu(float x, float mean, float std) {
	float coeff = 1.0f / (2.0f * M_PI * sqrtf(std));//ÎªÁËÆ½ºâ
	float exponent = expf(-0.5f * powf((x - mean) / std, 2));
	return coeff * exponent;
}

__device__ inline float mix_gaussian_gpu(float data, ActionPair* RF) {
	float r = 0;
	for (int i = 0; i < RF->num; i++) {
		r += gaussian_gpu(data, RF->means[i], RF->stds[i]);
	}
	return r;
}

__device__ inline float sin_af(float x) {
	return sinf(x * M_PI);
}