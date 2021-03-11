#include <torch/script.h>
#include <iostream>

#include "dump.hpp"
#include "network.hpp"


int main() {
    std::ifstream fin("/Users/maxkrutonog/Desktop/model", std::ios::binary | std::ios::in);
    dumpapi::network::Module network({
        new dumpapi::network::BatchNorm()
    });
    network.load_weight(fin);
    fin.close();
    fin.open("/Users/maxkrutonog/Desktop/tensor", std::ios::binary | std::ios::in);
    auto tensor = dumpapi::load_tensor(fin);
    fin.close();
    std::cout << network.forward(tensor) << std::endl;
}
