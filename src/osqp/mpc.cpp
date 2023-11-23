#include "OsqpEigen/OsqpEigen.h"
#include <Eigen/Dense>
#include <iostream>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/fmt/ostr.h"
#include <qpOASES.hpp>


/*
  row 行
  col 列
  预测时域Np = 10
  控制时域Nc = Np - 1 = 9
*/

// 自定义spdlog格式器，用于将Eigen矩阵格式化为字符串
template<typename Derived>
std::string eigenObjectToString(const Eigen::DenseBase<Derived>& eigenObject) {
    std::ostringstream stream;
    stream << eigenObject;
    return stream.str();
}

// 将Eigen矩阵或向量转换为real_t类型数组
template <typename EigenType>
void eigenToArray(const EigenType& data, qpOASES::real_t* array) {
    const qpOASES::real_t* data_ptr = data.data();
    size_t size = data.size();
    std::memcpy(array, data_ptr, size * sizeof(qpOASES::real_t));
}

// 辅助函数模板，将qpOASES::real_t类型的数组转换为字符串
template <typename T>
std::string arrayToString(const T& arr, std::size_t size) {
    std::ostringstream stream;
    for (std::size_t i = 0; i < size; ++i) {
        stream << arr[i] << " ";
    }
    return stream.str();
}

 
int main() {
  int Np = 10;
  int Nc = Np - 1;
  float v_min = 0;
  float v_max = 4;
  float w_min = -1;
  float w_max = 1;
  auto logger = spdlog::basic_logger_mt("mpc_logger", "logs/mpc-log.txt",true);
  logger->set_level(spdlog::level::info);
  Eigen::VectorXd init_state(3);  // 初始状态
  init_state << 0.3, 0.4, 0.3;  // 当前点位置
  
  logger->info("init_state: {}", eigenObjectToString(init_state));
  int state_dim = init_state.rows();
  Eigen::VectorXd ref_state(Np * state_dim);
  ref_state << 0.5, 0.3, 0.2,
               0.6, 0.4, 0.25,
               0.7, 0.45, 0.3,
               0.8, 0.5, 0.2,
               0.9, 0.6, 0.16,
               1.0, 0.68, 0.23,
               1.1, 0.74, 0.4,
               1.2, 0.8, 0.32,
               1.3, 0.82, 0.26,
               1.4, 0.9, 0.37;
  logger->info("ref_state: {}", eigenObjectToString(ref_state));
  
  Eigen::MatrixXd A;  // 状态系数矩阵
  A.resize(3, 3);
  A << 1.0, 0.0, -2 * 0.01 * std::sin(0.3),
       0.0, 1.0, 2 * 0.01 * std::cos(0.3),
       0.0, 0.0, 1.0;
       
  logger->info("A: {}", eigenObjectToString(A));
  Eigen::MatrixXd B;  // 控制系数矩阵
  B.resize(3, 2);
  B << 0.01 * std::cos(0.3), 0.0,
       0.01 * std::sin(0.3), 0.0, 
       0.0, 0.01;
  logger->info("B: {}", eigenObjectToString(B));
  
  int control_dim = B.cols();
  
  Eigen::MatrixXd Q;  // 代价函数状态调整矩阵
  Q.resize(state_dim, state_dim);
  Eigen::MatrixXd R;  // 代价函数控制调整矩阵
  R.resize(control_dim, control_dim);
  Q << 1.0, 0.0, 0.0,
       0.0, 1.0, 0.0,
       0.0, 0.0, 1.0;
  logger->info("Q: {}", eigenObjectToString(Q));
  R << 0.1, 0.0,
       0.0, 0.1;
  logger->info("R: {}", eigenObjectToString(R));
  Eigen::MatrixXd H_dense;  // 二次qp问题的hessian密集矩阵
  Eigen::SparseMatrix<double> H;
  H_dense.resize(Nc * control_dim, Nc * control_dim);
  H_dense.setZero();
  H.resize(Nc * control_dim, Nc * control_dim);
  H.setZero();
  Eigen::VectorXd f(Nc * control_dim);  // 二次qp问题的梯度矩阵

  Eigen::VectorXd lowerBound(control_dim);  // 约束：优化变量的下界
  Eigen::VectorXd upperBound(control_dim);  // 约束：优化变量的上界
  
  lowerBound << v_min, w_min;
  upperBound << v_max, w_max;
  logger->info("lowerBound: {}", eigenObjectToString(lowerBound));
  logger->info("upperBound: {}", eigenObjectToString(upperBound));
  
  Eigen::MatrixXd A_bar;  // 预测代价函数，整合A之后的矩阵
  Eigen::MatrixXd B_bar;  // 预测代价函数，整合A、B之后的矩阵
  Eigen::MatrixXd Q_bar; 
  Eigen::MatrixXd R_bar; 
  Eigen::MatrixXd linearMatrix_bar_dense;
  Eigen::SparseMatrix<double> linearMatrix_bar;
  Eigen::VectorXd lowerBound_bar(Nc * control_dim);
  Eigen::VectorXd upperBound_bar(Nc * control_dim);

  A_bar.resize(Np * state_dim, state_dim);
  B_bar.resize(Np * state_dim, Nc * control_dim);
  Q_bar.resize(Np * state_dim, Np * state_dim);
  R_bar.resize(Nc * control_dim, Nc * control_dim);
  
  B_bar.setZero();
  Q_bar.setZero();
  R_bar.setZero();
  
  
  linearMatrix_bar_dense.resize(Nc * control_dim, Nc * control_dim);
  linearMatrix_bar_dense.setZero();
  linearMatrix_bar_dense.setIdentity();

  int start_index = 0;  // 构造A_bar的起始点x坐标

  Eigen::MatrixXd tempB_bar;
  tempB_bar.resize(state_dim, (Nc + 1) * control_dim);
  tempB_bar.setZero();

  Eigen::MatrixXd temp_A;  // 储存A的幂次
  temp_A.resize(state_dim, state_dim);
  temp_A.setIdentity();

  // A_bar、B_bar的实现
  for (int i = 1; i <= Np; ++i) {
    Eigen::MatrixXd temp;
    temp.resize(state_dim, Nc * control_dim);
    temp << tempB_bar.block(0, 0, state_dim, B_bar.cols());
    tempB_bar << temp_A * B, temp;
    
    B_bar.block(start_index, 0, state_dim, B_bar.cols()) = 
      tempB_bar.block(0, 0, state_dim, tempB_bar.cols() - control_dim);
    temp_A *= A;
    A_bar.block(start_index, 0, state_dim, state_dim) = temp_A;
    start_index += state_dim;
  }
  
  logger->info("A_bar: {}", eigenObjectToString(A_bar));
  logger->info("B_bar: {}", eigenObjectToString(B_bar));

  for (int i = 0; i < Np; ++i) {
    Q_bar.block(state_dim * i, state_dim * i, state_dim, state_dim) = Q;
  }
  
  logger->info("Q_bar: {}", eigenObjectToString(Q_bar));
  for (int i = 0; i < Nc; ++i) {
    R_bar.block(i * control_dim, i * control_dim, control_dim, control_dim) = R;
    lowerBound_bar.block(i * control_dim, 0, control_dim, 1) = lowerBound;
    upperBound_bar.block(i * control_dim, 0, control_dim, 1) = upperBound;
  }
  
  logger->info("R_bar: {}", eigenObjectToString(R_bar));
  logger->info("lowerBound_bar: {}", eigenObjectToString(lowerBound_bar));
  logger->info("upperBound_bar: {}", eigenObjectToString(upperBound_bar));

  H_dense = B_bar.transpose() * Q_bar * B_bar;
  
  logger->info("H_dense: {}", eigenObjectToString(H_dense));
  f = B_bar.transpose() * Q_bar * A_bar * init_state - 
      B_bar.transpose() * Q_bar * ref_state;
  // std::cout << "uuuuuuuuu" << std::endl;
  logger->info("f: {}", eigenObjectToString(f));
  H = H_dense.sparseView();
  // logger->info("H: {}", eigenObjectToString(H));
  linearMatrix_bar = linearMatrix_bar_dense.sparseView();
  // logger->info("linearMatrix_bar: {}", eigenObjectToString(linearMatrix_bar));

  // instantiate the solver
  OsqpEigen::Solver solver;
 
  // settings
  solver.settings()->setVerbosity(false);
  solver.settings()->setWarmStart(true);
 
  // set the initial data of the QP solver
  solver.data()->setNumberOfVariables(Nc * control_dim);   //变量数n,linearMatrix_bar的列数
  solver.data()->setNumberOfConstraints(Nc * control_dim); //约束数m,linearMatrix_bar的行数
  if (!solver.data()->setHessianMatrix(H)) return 1;
  if (!solver.data()->setGradient(f)) return 1;
  if (!solver.data()->setLinearConstraintsMatrix(linearMatrix_bar)) return 1;
  if (!solver.data()->setLowerBound(lowerBound_bar)) return 1;
  if (!solver.data()->setUpperBound(upperBound_bar)) return 1;
 
  // instantiate the solver
  if (!solver.initSolver()) return 1;
 
  Eigen::VectorXd QPSolution_control;
 
  // solve the QP problem
  // if (!solver.solveProblem()) {
  //   return 1;
  // }
  solver.solveProblem();
 
  QPSolution_control = solver.getSolution();
  logger->info("QPSolution_control: {}", eigenObjectToString(QPSolution_control));
  std::cout << "QPSolution" << std::endl
            << QPSolution_control[0] << "  " << QPSolution_control[1] << std::endl; 
  


  // qpOASES求解
  qpOASES::QProblem mpc_qp(Nc * control_dim, 0);
  
  qpOASES::int_t nWSR = 1000;
  qpOASES::real_t* HArray = new qpOASES::real_t[Nc * control_dim * Nc * control_dim];
  qpOASES::real_t* fArray = new qpOASES::real_t[Nc * control_dim];
  qpOASES::real_t* lowerBoundArray = new qpOASES::real_t[Nc * control_dim];
  qpOASES::real_t* upperBoundArray = new qpOASES::real_t[Nc * control_dim];
  qpOASES::real_t* linearMatrixArray = new qpOASES::real_t[Nc * control_dim * Nc * control_dim];

  eigenToArray(H_dense, HArray);
  eigenToArray(f, fArray);
  eigenToArray(lowerBound_bar, lowerBoundArray);
  eigenToArray(upperBound_bar, upperBoundArray);
  eigenToArray(linearMatrix_bar_dense, linearMatrixArray);
  logger->info("HArray: {}", arrayToString(HArray, Nc * control_dim * Nc * control_dim));
  logger->info("fArray: {}", arrayToString(fArray, Nc * control_dim));
  logger->info("lowerBoundArray: {}", arrayToString(lowerBoundArray, Nc * control_dim));
  logger->info("upperBoundArray: {}", arrayToString(upperBoundArray, Nc * control_dim));
  logger->info("linearMatrixArray: {}", arrayToString(linearMatrixArray, Nc * control_dim * Nc * control_dim));
  // mpc_qp.init(HArray, fArray, linearMatrixArray, NULL, NULL,
  //              lowerBoundArray, upperBoundArray, nWSR);
  mpc_qp.init(HArray, fArray, NULL, 
              lowerBoundArray, upperBoundArray, NULL, NULL, nWSR);

  qpOASES::real_t UOpt[Nc * control_dim];
  mpc_qp.getPrimalSolution(UOpt);
  // std::cout << "UOpt size" << sizeof(UOpt) / sizeof(UOpt[0]) << std::endl;
  std::cout<< "\nUOpt1 = [" << UOpt[0] << ", " << UOpt[1] << "]" << std::endl;
  std::cout<< "UOpt2 = [" << UOpt[2] << ", " << UOpt[3] << "]" << std::endl;
  std::cout<< "UOpt3 = [" << UOpt[4] << ", " << UOpt[5] << "]" << std::endl;
  std::cout<< "UOpt4 = [" << UOpt[6] << ", " << UOpt[7] << "]" << std::endl;
  std::cout<< "UOpt5 = [" << UOpt[8] << ", " << UOpt[9] << "]" << std::endl;
  std::cout<< "UOpt6 = [" << UOpt[10] << ", " << UOpt[11] << "]" << std::endl;
  std::cout<< "UOpt7 = [" << UOpt[12] << ", " << UOpt[13] << "]" << std::endl;
  std::cout<< "UOpt8 = [" << UOpt[14] << ", " << UOpt[15] << "]" << std::endl;
  std::cout<< "UOpt9 = [" << UOpt[16] << ", " << UOpt[17] << "]" << std::endl;
  std::cout << "ObjVal: " << mpc_qp.getObjVal() << std::endl; // 目标函数值
  delete[] HArray;
  delete[] fArray;
  delete[] lowerBoundArray;
  delete[] upperBoundArray;
  delete[] linearMatrixArray;











  spdlog::drop_all();
  return 0;
}
