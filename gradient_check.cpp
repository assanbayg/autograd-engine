#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>

#include "autograd/engine.hpp"

bool check_gradient(const std::string& op, double data,
                    std::function<ValuePtr(ValuePtr)> autograd_op,
                    std::function<double(double)> num_op, double h = 1e-6,
                    double tolerance = 1e-7) {
  ValuePtr x = Value::make(data);
  ValuePtr y = autograd_op(x);
  y->backward();
  double analyt_grad = x->grad;

  double f_plus = num_op(data + h);
  double f_minus = num_op(data - h);
  double num_grad = (f_plus - f_minus) / (2.0 * h);

  double abs_diff = std::abs(analyt_grad - num_grad);
  double rel_error =
      abs_diff / std::max(1e-8, (std::abs(analyt_grad) + std::abs(num_grad)));

  std::cout << "GRADIENT CHECK" << op << " at x = " << data << "\n";
  std::cout << "Analytical:     " << analyt_grad << "\n";
  std::cout << "Numerical:      " << num_grad << "\n";
  std::cout << "Relative Error: " << rel_error << "\n";

  if (rel_error < tolerance) {
    std::cout << "PASSED!\n\n";
    return true;
  } else {
    std::cout << "FAILED!!!\n\n";
    return false;
  }
}