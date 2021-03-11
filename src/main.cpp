#include <torch/script.h>
#include <iostream>

#include "dump.hpp"
#include "network.hpp"


int main() {
    std::ifstream fin("/Users/maxkrutonog/Desktop/model", std::ios::binary | std::ios::in);
    dumpapi::network::Module network({
        new dumpapi::network::Linear(true),
        new dumpapi::network::Linear(true),
    });
    network.load_weight(fin);
    std::cout << network.forward(torch::zeros({1, 3})) << std::endl;
    fin.close();
}
