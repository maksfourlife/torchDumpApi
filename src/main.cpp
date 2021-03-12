#include <torch/script.h>
#include <iostream>

#include "dump.hpp"
#include "network.hpp"


int main() {
    std::ifstream fin("/Users/maxkrutonog/Desktop/model", std::ios::binary | std::ios::in);
    dumpapi::network::Module network({
        new dumpapi::network::ConvBNActivation(2, 1),
        new dumpapi::network::InvertedResidual(false, 1, 1, 32),
        new dumpapi::network::InvertedResidual(false, 1, 0, 1, 2, 1, 96),
        new dumpapi::network::InvertedResidual(true, 1, 0, 1, 1, 1, 144),
        new dumpapi::network::InvertedResidual(false, 1, 0, 1, 2, 1, 144),
        new dumpapi::network::InvertedResidual(true, 1, 0, 1, 1, 1, 192),
        new dumpapi::network::InvertedResidual(true, 1, 0, 1, 1, 1, 192),
        new dumpapi::network::InvertedResidual(false, 1, 0, 1, 2, 1, 192),
        new dumpapi::network::InvertedResidual(true, 1, 0, 1, 1, 1, 384),
        new dumpapi::network::InvertedResidual(true, 1, 0, 1, 1, 1, 384),
        new dumpapi::network::InvertedResidual(true, 1, 0, 1, 1, 1, 384), // 10
        new dumpapi::network::InvertedResidual(false, 1, 0, 1, 1, 1, 384),
        new dumpapi::network::InvertedResidual(true, 1, 0, 1, 1, 1, 576),
        new dumpapi::network::InvertedResidual(true, 1, 0, 1, 1, 1, 576),
        new dumpapi::network::InvertedResidual(false, 1, 0, 1, 2, 1, 576), // 14
        new dumpapi::network::InvertedResidual(true, 1, 0, 1, 1, 1, 960),
        new dumpapi::network::InvertedResidual(true, 1, 0, 1, 1, 1, 960),
        new dumpapi::network::InvertedResidual(false, 1, 0, 1, 1, 1, 960),
        new dumpapi::network::ConvBNActivation(),
    });
    network.load_weight(fin);
    fin.close();
    std::cout << network.forward(torch::linspace(0, 1, 7500).reshape({1, 3, 50, 50})).index({0, 1}) << std::endl;
}
