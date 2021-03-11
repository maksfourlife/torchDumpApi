#include "network.hpp"
#include <type_traits>

namespace dumpapi {
namespace network {

class Sequential : Callable {
public:
    Sequential(std::vector<Module> children) {
        this->children = children;
    }
    // void load_weight(std::ifstream fin) {
    //     for (auto &child : children) {
    //         if (std::is_base_of<decltype(child), Module>())
    //             child->load_weight();
    //     }
    // }
private:
    std::vector<Module> children;
};

};
};
