#include "DefaultPara.h"
#include <chrono>

std::string getCurrentTimeString() {
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char buffer[80];
    std::strftime(buffer, sizeof(buffer), "%Y.%m.%d_%H-%M-%S", std::localtime(&now));
    return std::string(buffer);
}

ofstream get_recorder() {
    std::string fileName = "./data_/" + getCurrentTimeString() + ".dat";
    ofstream r(fileName, std::ios::out | std::ios::binary);
    auto i = r.is_open();
    return r;
}

int ACTION_PAIR_NUM = 3;
int STEP_ACTION_PAIR_NUM = 2;
int KERNEL_PAIR_NUM = 3;
int DYNAMIC_LEVEL = 2;
float CELL_BELONG_RADIUS = 5;
int ENV_SIZE = 500;
int KERNEL_LENGTH = 5;
int CHANNEL = 3;
float DELTA_T = 0.1;

float* D_GENE_WEIGHT = 0;
float* D_GENE_FCL_MATRIX = 0;
float* D_G_STEP_AP_MEAN = 0;
float* D_G_BORN_AP_MEAN = 0;
float* D_G_DEATH_AP_MEAN = 0;
float* D_G_STEP_AP_STD = 0;
float* D_G_BORN_AP_STD = 0;
float* D_G_DEATH_AP_STD = 0;
float* D_G_DYNAMIC_A = 0;
float* D_G_DYNAMIC_PHI = 0;
float* D_G_KERNEL_AP_MEAN = 0;
float* D_G_KERNEL_AP_STD = 0;
extern ofstream recorder = get_recorder();
extern float D_G_BASE=0;
extern float D_G_BASE_K=0;

