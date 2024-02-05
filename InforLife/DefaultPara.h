#pragma once

extern int ACTION_PAIR_NUM;
extern int STEP_ACTION_PAIR_NUM;
extern int KERNEL_PAIR_NUM;
extern int DYNAMIC_LEVEL;
extern float CELL_BELONG_RADIUS;
extern int ENV_SIZE;
extern int KERNEL_LENGTH;
extern int CHANNEL;
extern float DELTA_T;
constexpr auto PI = 3.14159265358979323846;
constexpr int GENE_PLACE_NUM = 8;

extern float* D_GENE_WEIGHT;
extern float* D_GENE_FCL_MATRIX;
extern float* D_G_STEP_AP_MEAN;
extern float* D_G_BORN_AP_MEAN;
extern float* D_G_DEATH_AP_MEAN;
extern float* D_G_STEP_AP_STD;
extern float* D_G_BORN_AP_STD;
extern float* D_G_DEATH_AP_STD;
extern float* D_G_DYNAMIC_A;
extern float* D_G_DYNAMIC_PHI;
extern float* D_G_KERNEL_AP_MEAN;
extern float* D_G_KERNEL_AP_STD;