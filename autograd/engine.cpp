#include "engine.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>

ValuePtr Value::pow(double exponent) {
  auto out = Value::make(std::pow(this->data, exponent),
                         std::unordered_set{shared_from_this()}, "pow");

  out->_backward = [self = shared_from_this(), exponent](double grad) {
    self->grad += exponent * std::pow(self->data, exponent - 1) * grad;
  };

  return out;
}

ValuePtr Value::relu() {
  auto newData = this->data < 0 ? 0.0 : this->data;
  auto out =
      Value::make(newData, std::unordered_set{shared_from_this()}, "ReLU");

  out->_backward = [self = shared_from_this(), newData](double grad) {
    self->grad += newData > 0 ? grad : 0.0;
  };

  return out;
}

void Value::backward() {
  // TODO !!!
}

void Value::print(std::ostream& out) const {}

ValuePtr operator+(const ValuePtr& a, const ValuePtr& b) {
  auto out = Value::make(a->data + b->data, std::unordered_set{a, b}, "+");

  out->_backward = [a, b](double grad) {
    a->grad += grad;
    b->grad += grad;
  };

  return out;
}

ValuePtr operator*(const ValuePtr& a, const ValuePtr& b) {
  auto out = Value::make(a->data * b->data, std::unordered_set{a, b}, "*");

  out->_backward = [a, b](double grad) {
    a->grad += b->data * grad;
    b->grad += a->data * grad;
  };

  return out;
}

ValuePtr operator-(const ValuePtr& a) {  // Negation
  auto out = Value::make(-a->data, std::unordered_set{a}, "neg");

  out->_backward = [a](double grad) { a->grad += -grad; };

  return out;
}

ValuePtr operator-(const ValuePtr& a, const ValuePtr& b) { return a + (-b); }

ValuePtr operator/(const ValuePtr& a, const ValuePtr& b) {
  return a * (b->pow(-1));
}

std::ostream& operator<<(std::ostream& os, const ValuePtr& val) {
  if (val) {
    val->print(os);
  }
  return os;
}