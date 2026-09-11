#include "sampling.h"

#include <cassert>
#include <cmath>
#include <limits>

int main() {
    float first_step[] = {1.0F, 100.0F, 2.0F};
    suppress_initial_eos(first_step, 3, 1, 0);
    assert(std::isinf(first_step[1]) && first_step[1] < 0.0F);
    assert(first_step[0] == 1.0F);
    assert(first_step[2] == 2.0F);

    float later_step[] = {1.0F, 100.0F, 2.0F};
    suppress_initial_eos(later_step, 3, 1, 1);
    assert(later_step[1] == 100.0F);

    float invalid_id[] = {1.0F, 2.0F};
    suppress_initial_eos(invalid_id, 2, std::numeric_limits<int>::max(), 0);
    assert(invalid_id[0] == 1.0F && invalid_id[1] == 2.0F);
    return 0;
}
