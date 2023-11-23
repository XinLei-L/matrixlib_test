#include <iostream>
#include <Eigen/Dense>
#include <vector>

typedef double real_t;  // 或根据需要使用float

// 将Eigen矩阵或向量转换为real_t类型数组
template <typename EigenType>
void eigenToArray(const EigenType& data, real_t* array) {
    const real_t* data_ptr = data.data();
    size_t size = data.size();
    std::memcpy(array, data_ptr, size * sizeof(real_t));
}

int main() {
    // 使用Eigen矩阵
    Eigen::MatrixXd eigenMatrix(3, 3);
    eigenMatrix << 1, 2, 3,
                   4, 5, 6,
                   7, 8, 9;

    size_t eigenSize = eigenMatrix.size();
    real_t* eigenArray = new real_t[eigenSize];
    eigenToArray(eigenMatrix, eigenArray);

    // 打印Eigen矩阵数组元素
    std::cout << "Eigen矩阵数组: ";
    for (size_t i = 0; i < eigenSize; ++i) {
        std::cout << eigenArray[i] << " ";
    }
    std::cout << std::endl;

    delete[] eigenArray;

    // 使用Eigen向量
    Eigen::VectorXd eigenVector(4);
    eigenVector << 1, 2, 3, 4;

    size_t vectorSize = eigenVector.size();
    real_t* vectorArray = new real_t[vectorSize];
    eigenToArray(eigenVector, vectorArray);

    // 打印Eigen向量数组元素
    std::cout << "Eigen向量数组: ";
    for (size_t i = 0; i < vectorSize; ++i) {
        std::cout << vectorArray[i] << " ";
    }
    std::cout << std::endl;

    delete[] vectorArray;

    return 0;
}
