#include "engine.hpp"
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
  auto x1w1 = Value::make(x1->data * w1->data, std::unordered_set{x1, w1}, "*");
  auto x2w2 = Value::make(x2->data * w2->data, std::unordered_set{x2, w2}, "*");

  auto sum =
      Value::make(x1w1->data + x2w2->data, std::unordered_set{x1w1, x2w2}, "+");
  auto logit =
      Value::make(sum->data + b->data, std::unordered_set{sum, b}, "+");

  // Activation function
  auto out = Value::make(0.9999, std::unordered_set{logit}, "tanh");

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

  std::cout << "Finished!" << std::endl;
  return 0;
}