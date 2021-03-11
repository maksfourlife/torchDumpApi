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
    Module(std::vector<Module> children);
    virtual at::Tensor forward(at::Tensor input);
    void load_weight(std::ifstream& fin);
protected:
    std::vector<at::Tensor> weights;
    std::vector<Module> children;
    int n_weights;
};

class Linear : public Module {
public:
    Linear(bool bias = true);
    virtual at::Tensor forward(at::Tensor input);
private:
    bool bias;
};

};
};
