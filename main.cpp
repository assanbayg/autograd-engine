#include "autograd/engine.hpp"
#include "gradient_check.cpp"
#include "trace_graph.cpp"

int main(int argc, char* argv[]) {
  std::cout << "Drawing dummy graph..." << std::endl;

  auto x1 = Value::make(-1.0);
  Value::make(-1.0);
  auto x2 = Value::make(2.0);
  auto w1 = Value::make(3.0);
  auto w2 = Value::make(1.5);

  auto b = Value::make(0.5);

  // Forward Pass: x1*w1 + x2*w2 + b
  auto x1w1 = Value::make(x1->data * w1->data, Children{x1, w1}, "*");
  auto x2w2 = Value::make(x2->data * w2->data, Children{x2, w2}, "*");

  auto sum = Value::make(x1w1->data + x2w2->data, Children{x1w1, x2w2}, "+");
  auto logit = Value::make(sum->data + b->data, Children{sum, b}, "+");

  // Activation function
  auto out = Value::make(0.9999, Children{logit}, "tanh");

  // Backward Pass
  out->grad = 1.0;
  logit->grad = 0.5;
  b->grad = 0.5;
  sum->grad = 0.5;
  x1w1->grad = 0.5;
  x2w2->grad = 0.5;
  x1->grad = 1.5;
  w1->grad = -0.5;
  x2->grad = 0.75;
  w2->grad = 1.0;

  // Draw the graph!
  drawDot(out, "LR");

  std::cout << "Finished drawing the graph!" << std::endl;

  std::cout << "Testing autograd engine\n\n" << std::endl;

  check_gradient(
      "tanh", 0.5, [](ValuePtr v) { return v->tanh(); },
      [](double d) { return std::tanh(d); });

  check_gradient(
      "reLu", 2.0, [](ValuePtr v) { return v->relu(); },
      [](double d) { return d < 0 ? 0.0 : d; });

  check_gradient(
      "pow (x^3)", 1.5, [](ValuePtr v) { return v->pow(3.0); },
      [](double d) { return std::pow(d, 3.0); });

  check_gradient(
      "x*x", 1.5, [](ValuePtr v) { return v * v; },
      [](double d) { return d * d; });

  check_gradient(
      "x+x", 1.5, [](ValuePtr v) { return v + v; },
      [](double d) { return d + d; });

  check_gradient(
      "x*x + x", 1.5, [](ValuePtr v) { return v * v + v; },
      [](double d) { return d * d + d; });

  return 0;
}