// osqp-eigen
#include "OsqpEigen/OsqpEigen.h"
 
// eigen
#include <Eigen/Dense>
#include <iostream>
 
int main() {
    // allocate QP problem matrices and vectores
    Eigen::SparseMatrix<double> hessian(2, 2);      //P: n*n正定矩阵,必须为稀疏矩阵SparseMatrix
    Eigen::VectorXd gradient(2);                    //Q: n*1向量
    Eigen::SparseMatrix<double> linearMatrix(2, 2); //A: m*n矩阵,必须为稀疏矩阵SparseMatrix
    Eigen::VectorXd lowerBound(2);                  //L: m*1下限向量
    Eigen::VectorXd upperBound(2);                  //U: m*1上限向量
 
    hessian.insert(0, 0) = 2.0; //注意稀疏矩阵的初始化方式,无法使用<<初始化
    hessian.insert(1, 1) = 2.0;
    // std::cout << "hessian:" << std::endl
    //           << hessian << std::endl;
    gradient << -2, -2;
    linearMatrix.insert(0, 0) = 1.0; //注意稀疏矩阵的初始化方式,无法使用<<初始化
    linearMatrix.insert(1, 1) = 1.0;
    // std::cout << "linearMatrix:" << std::endl
    //           << linearMatrix << std::endl;
    lowerBound << 0, 0;
    upperBound << 1.5, 1.5;
 
    // instantiate the solver
    OsqpEigen::Solver solver;
 
    // settings
    solver.settings()->setVerbosity(false);
    /*
        这行代码设置求解器的输出详细程度。setVerbosity 函数用于设置求解器的冗余级别，即是否输出冗余信息。
        参数 false 表示关闭冗余信息的输出，即在求解过程中不输出详细的调试信息。
        如果需要在求解过程中查看详细信息，可以将参数设置为 true
    */
    solver.settings()->setWarmStart(true);
    /*
        这行代码设置是否使用热启动（warm start）。
        热启动是一种优化问题求解中的技术，它允许在前一次求解的基础上进行新一轮的求解，以加速收敛速度。
        通过设置 setWarmStart 函数的参数为 true，表示启用热启动功能。
        这通常在需要多次迭代求解相似问题时提高求解效率。
    */
 
    // set the initial data of the QP solver
    solver.data()->setNumberOfVariables(2);   //变量数n
    solver.data()->setNumberOfConstraints(2); //约束数m
    if (!solver.data()->setHessianMatrix(hessian))
        return 1;
    if (!solver.data()->setGradient(gradient))
        return 1;
    if (!solver.data()->setLinearConstraintsMatrix(linearMatrix))
        return 1;
    if (!solver.data()->setLowerBound(lowerBound))
        return 1;
    if (!solver.data()->setUpperBound(upperBound))
        return 1;
 
    // instantiate the solver
    if (!solver.initSolver())
        return 1;
 
    Eigen::VectorXd QPSolution;
 
    // solve the QP problem
    if (!solver.solve()) {
        return 1;
    }
 
    QPSolution = solver.getSolution();
    std::cout << "QPSolution" << std::endl
              << QPSolution << std::endl; //输出为m*1的向量
    return 0;
}
