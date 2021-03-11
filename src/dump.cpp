#include "dump.hpp"


template <typename T>
T from_stream(std::ifstream& fin, char* buf) {
    T n;
    fin.read(buf, sizeof(T));
    std::memcpy((void*)&n, buf, sizeof(T));
    return n;
}

template <typename T>
T prod(std::vector<T> vec) {
    T p = vec[0];
    for (size_t i = 1; i < vec.size(); i++)
        p *= vec[i];
    return p;
}

namespace dumpapi {

std::vector<int64_t> load_sizes(std::ifstream& fin) {
    char buf[sizeof(int)];
    std::vector<int64_t> size(from_stream<int>(fin, buf));
    for (int i = 0; i < size.size(); i++)
        size[i] = (int64_t)from_stream<int>(fin, buf);
    return size;
}

at::Tensor load_tensor(std::ifstream& fin) {
    auto size = load_sizes(fin);
    char buf[sizeof(float)];
    std::vector<float> content(prod<int64_t>(size));
    for (int64_t i = 0; i < content.size(); i++)
        content[i] = from_stream<float>(fin, buf);
    auto shape = new c10::ArrayRef<int64_t>(size.data(), size.size());
    return torch::tensor(content).reshape(*(c10::IntArrayRef*)shape);
}

};
