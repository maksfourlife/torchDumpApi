#include <torch/script.h>
#include <iostream>

#include "dump.hpp"
#include "network.hpp"


int main() {
    std::ifstream fin("/Users/maxkrutonog/Desktop/model", std::ios::binary | std::ios::in);
    dumpapi::network::Module network({
        new dumpapi::network::ConvBNActivation(2, 1),
        new dumpapi::network::InvertedResidual(false, 1, 1, 32)
    });
    network.load_weight(fin);
    fin.close();
    std::cout << network.forward(torch::linspace(0, 1, 75).reshape({1, 3, 5, 5})).index({0, 1}) << std::endl;
}
