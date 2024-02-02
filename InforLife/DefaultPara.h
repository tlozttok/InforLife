#pragma once

int ACTION_PAIR_NUM = 3;
int STEP_ACTION_PARI_NUM = 2;
int KERNEL_PAIR_NUM = 3;
int DYNAMIC_LEVEL = 2;
float CELL_BELONG_RADIUS = 30;
int ENV_SIZE = 2000;
int KERNEL_LENGTH = 5;
int CHANNEL = 3;
float DELTA_T = 0.1;
constexpr auto PI = 3.14159265358979323846;
constexpr int GENE_PLACE_NUM = 8;

float D_GENE_WEIGHT[] = {1/3,1/3,1/3};
float D_GENE_FCL_MATRIX[] = {1,0,0,0,1,0,0,0,1};
float D_G_STEP_AP_MEAN[] = { 0.3,0.7 };
float D_G_BORN_AP_MEAN[] = { 0.1,0.3,0.6 };
float D_G_DEATH_AP_MEAN[] = { 0.25,0.5,0.7 };
float D_G_STEP_AP_STD[] = { 0.01,0.02 };
float D_G_BORN_AP_STD[] = { 0.02,0.04,0.02 };
float D_G_DEATH_AP_STD[] = { 0.04,0.02,0.04 };
float D_G_DYNAMIC_A[] = { 1,0.5 };
float D_G_DYNAMIC_PHI[] = { 0,PI };
float D_G_KERNEL_AP_MEAN[] = {0.1,0.3,0.6};
float D_G_KERNEL_AP_STD[] = { 0.04,0.03,0.02 };