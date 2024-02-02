#pragma once
#include <mutex>
using std::mutex;

static mutex cell_territory_lock;
static mutex dynamic_lock;
static mutex gpu_data_lock;