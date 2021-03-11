#include <torch/script.h>
#include <vector>
#include <fstream>

// SPEC:
// [lensize][size ...][data]

template <typename T>
T from_stream(std::ifstream& fin, char* buf);

std::vector<int64_t> get_sizes(std::ifstream& fin);

template <typename T>
T prod(std::vector<T> vec);

at::Tensor get_content(std::ifstream& fin);
