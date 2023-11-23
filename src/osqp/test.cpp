// osqp-eigen
#include "OsqpEigen/OsqpEigen.h"
 
// eigen
#include <Eigen/Dense>
#include <iostream>

 
int main()
{  
  Eigen::VectorXf init_state(3);  // VectorXf是列向量
  Eigen::VectorXf lowerBound_bar(3 * init_state.rows());
  
  init_state << 0.3, 0.4, 0.3;  // 当前点位置
  for (int i = 0; i < 3; ++i) {
    lowerBound_bar.block(i * init_state.rows(), 0, init_state.rows(), 1) = init_state;
  }

  Eigen::MatrixXf Q; 
  Q.resize(3, 3);
  Q.setIdentity();
  
  std::cout << "init_state_cols: " << init_state.cols() << std::endl;
  std::cout << "init_state_rows: " << init_state.rows() << std::endl;

  std::cout << "Q: " << std::endl;
  std::cout << Q << std::endl;

  std::cout << "Q.block: " << std::endl;
  std::cout << Q.block(0, 0, 2, 1) << std::endl;

  std::cout << "lowerBound_bar: " << std::endl;
  std::cout << lowerBound_bar << std::endl;
  
 
  return 0;
}
