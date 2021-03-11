#include <torch/script.h>
#include <vector>
#include <fstream>

// SPEC:
// [lensize][size ...][data]

namespace dumpapi {

std::vector<int64_t> load_sizes(std::ifstream& fin);

at::Tensor load_tensor(std::ifstream& fin);

};