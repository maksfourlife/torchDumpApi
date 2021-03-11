#include <torch/script.h>
#include <iostream>

#include "dump.hpp"


int main() {
    std::ifstream fin;
    fin.open("/Users/maxkrutonog/Desktop/tensor", std::ios::binary | std::ios::in);
    std::cout << dumpapi::load_tensor(fin) << std::endl;
    fin.close();
}
