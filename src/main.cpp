#include <torch/script.h>
#include <iostream>

#include "dump.hpp"


int main() {
    std::ifstream fin;
    fin.open("/Users/maxkrutonog/Desktop/tensor", std::ios::binary | std::ios::in);
    std::cout << get_content(fin) << std::endl;
    fin.close();
}
