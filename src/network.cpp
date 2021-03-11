#include "network.hpp"
#include "dump.hpp"

namespace dumpapi {
namespace network {

Module::Module() {
    this->n_weights = 0;
}

Module::Module(std::vector<Module*> children) : Module() {
    this->children = children;
}

void Module::load_weight(std::ifstream& fin) {
    if (this->children.size())
        for (auto& child : this->children)
            child->load_weight(fin);
    else {
        for (int i = 0; i < this->n_weights; i++)
            this->weights.push_back(dumpapi::load_tensor(fin));
    }
}

// does not implement inner ops
at::Tensor Module::forward(at::Tensor input) {
    auto output = input;
    std::cout << "in_module_forward\n";
    if (this->children.size())
        for (auto& child : this->children)
            output = child->forward(output);
    return output;
}

Linear::Linear(bool bias) : Module() {
    this->bias = bias;
    this->n_weights = 1 + bias;
}

at::Tensor Linear::forward(at::Tensor input) {
    if (this->bias)
        return torch::linear(input, this->weights[0], this->weights[1]);
    return torch::linear(input, this->weights[0]);
}

};
};
