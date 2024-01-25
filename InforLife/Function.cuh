#include "Background.cuh"
#include <corecrt_math_defines.h>

__device__ inline float gaussian(float x, float mean, float std) {
	float coeff = 1.0f / (std * sqrtf(2.0f * M_PI));
	float exponent = expf(-0.5f * powf((x - mean) / std, 2));
	return coeff * exponent;
}

__device__ inline float mix_gaussian(float data, ActionPair RF) {
	float r = 0;
	for (int i = 0; i < RF.num; i++) {
		r += gaussian(data, RF.means[i], RF.stds[i]);
	}
}