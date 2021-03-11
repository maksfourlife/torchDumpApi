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

class Module {
public:
    Module();
    Module(std::vector<Module*> children);
    virtual at::Tensor forward(at::Tensor input);
    void load_weight(std::ifstream& fin);
    std::vector<at::Tensor> weights;
    std::vector<Module*> children;
    int n_weights;
};

class Linear : public Module {
public:
    Linear(bool bias = true);
    at::Tensor forward(at::Tensor input) override;
    bool bias;
};

class Conv2d : public Module {
public:
    Conv2d(bool bias = false, int stride = 1, int padding = 0, int dilation = 1, int groups = 1);
    at::Tensor forward(at::Tensor input) override;
    bool bias;
    int stride;
    int padding;
    int dilation;
    int groups;
};

class BatchNorm : public Module {
public:
    BatchNorm(double eps = 1e-5, double momentum = 0.1);
    at::Tensor forward(at::Tensor input) override;
    double eps;
    double momentum;
};

};
};
