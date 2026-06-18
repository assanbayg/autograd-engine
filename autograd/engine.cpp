#include "engine.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>

ValuePtr Value::pow(double exponent) { return nullptr; }

ValuePtr Value::relu() { return nullptr; }

void Value::backward() {}

void Value::print(std::ostream& out) const {}

ValuePtr operator+(const ValuePtr& a, const ValuePtr& b) { return nullptr; }

ValuePtr operator*(const ValuePtr& a, const ValuePtr& b) { return nullptr; }

ValuePtr operator-(const ValuePtr& a, const ValuePtr& b) { return nullptr; }

ValuePtr operator/(const ValuePtr& a, const ValuePtr& b) { return nullptr; }

ValuePtr operator-(const ValuePtr& a) {  // Negation
  return nullptr;
}

std::ostream& operator<<(std::ostream& os, const ValuePtr& val) {
  if (val) {
    val->print(os);
  }
  return os;
}