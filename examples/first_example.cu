#include <vector>

// just include the relative path of the ridge source
#include "../src/ridge.cu"

int main(int argc, char const *argv[])
{
    std::vector<double> rhs = {1, 4, 2, 2, 5, 1, 3, 6, 1, 1, 2, 3};
    std::vector<double> lhs = {6, 15, 4};
    int const m = 3;
    int const n = 4;
    double const lamda = 0;

    std::vector<double> solution = cuRidgeSolve(rhs, lhs, m, n, lamda);

    // output should be 3.31561, -3.63121, 3.60506
    std::cout << solution[0] << ", " << solution[1] << ", " << solution[2] << std::endl;
    return 0;
}
