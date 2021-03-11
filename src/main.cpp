#include <torch/script.h>
#include <iostream>

#include "dump.hpp"
#include "network.hpp"


int main() {
    std::ifstream fin("/Users/maxkrutonog/Desktop/layer", std::ios::binary | std::ios::in);
    std::vector<dumpapi::network::Module*> children = {
        new dumpapi::network::Linear(true)
    };
    dumpapi::network::Module network(children);
    network.load_weight(fin);
    std::cout << network.children[0]->forward(torch::zeros({1, 3})) << std::endl;
    fin.close();
}
