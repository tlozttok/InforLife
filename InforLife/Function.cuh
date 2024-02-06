#include "Background.cuh"
#include <corecrt_math_defines.h>

__device__ inline float gaussian_gpu(float x, float mean, float std) {
	float coeff = 1.0f / (2.0f * M_PI * sqrtf(abs(std)));//ÎªÁËÆ½ºâ
	float exponent = expf(-0.5f * powf((x - mean) / std, 2));
	
	return coeff * exponent * (std / abs(std));
}

__device__ inline float mix_gaussian_gpu(float data, ActionPair* RF) {
	float r = 0;
	for (int i = 0; i < RF->num; i++) {
		float t = gaussian_gpu(data, RF->means[i], RF->stds[i]);
		r += t;
	}
	return r;
}

__device__ inline float sin_af(float x) {
	return sinf(x * M_PI);
}

inline float cut(float x) {
	if (x < 0) {
		return 0;
	}
	else if (x > 1) {
		return 1;
	}
	else return x;
}

__device__ inline float sigmoid_g(float x) {
	return 1 / (1 + pow(M_E, -3 * (x - 0.5)));
}

inline float sigmoid(float x) {
	return 1 / (1 + pow(M_E, -3 * (x - 0.5)));
}