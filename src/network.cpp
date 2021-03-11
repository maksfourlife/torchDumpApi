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

Conv2d::Conv2d(bool bias, int stride, int padding, int dilation, int groups) : Module() {
    this->bias = bias;
    this->n_weights = 1 + bias;
    this->stride = stride;
    this->padding = padding;
    this->dilation = dilation;
    this->groups = groups;
}

at::Tensor Conv2d::forward(at::Tensor input) {
    c10::optional<at::Tensor> bias = {};
    if (this->bias)
        bias = this->weights[1];
    return torch::conv2d(input, this->weights[0], bias, this->stride,
        this->padding, this->dilation, this->groups);
}

BatchNorm::BatchNorm(double eps, double momentum) {
    this->eps = eps;
    this->momentum = momentum;
    this->n_weights = 4;
}

at::Tensor BatchNorm::forward(at::Tensor input) {
    return torch::batch_norm(input, this->weights[0], this->weights[1], this->weights[2], this->weights[3],
        false, this->momentum, this->eps, false);
}

};
};
