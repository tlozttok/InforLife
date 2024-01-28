#pragma once
#include <mutex>
using std::mutex;

mutex cell_territory_lock;
mutex dynamic_lock;
mutex gpu_data_lock;