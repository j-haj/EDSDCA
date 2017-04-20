
#include "edsdca/memory/memory.h"

double* edsdca::memory::ConvertEigenMatrixToPtr(Eigen::MatrixXd& m) {
  double* c_m = (double*)malloc(sizeof(double) * m.size());
  Map<Eigen::MatrixXd>(c_m, m.rows(), m.cols()) = m; 
  return c_m;
}

double* edsdca::memory::ConvertEigenVectorToPtr(Eigen::VectorXd& v) {
  double *c_v = (double*)malloc(sizeof(double) * v.size());
  Map<Eigen::VectorXd>(c_v, v.size()) = v;
  return c_v;
}

Eigen::MatrixXd edsdca::memory::ConvertPtrToEigenMatrix(double* c, long row, long col) {
  return Map<MatrixXd>(c, row, col);
}

Eigen::VectorXd edsdca::memory::ConvertPtrToEigenVector(double* c, long row) {
  return Map<VectorXd>(c, row);
}
