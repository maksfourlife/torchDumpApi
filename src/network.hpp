#include <torch/script.h>
#include <fstream>
#include <vector>

// SPEC
// all modules should have children field (some modules have [] children)
// all modules whould have weights (-||-)
// all modules should have forward()
// all modules should have load_weight() which then call load_weight() of children


namespace dumpapi {
namespace network {

class Callable {
public:
    at::Tensor forward(at::Tensor input);
    void load_weight(std::ifstream& fin);
};

class Sequential : Callable {
public:
    Sequential(std::vector<Module> children);
private:
    std::vector<Module> children;
};

class Module : Callable {
private:
    std::vector<at::Tensor> weights;
};

class Linear : Module {};

};
};
