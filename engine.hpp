#ifndef VALUE_H_
#define VALUE_H_

#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>

class Value;
using ValuePtr = std::shared_ptr<Value>;

class Value : public std::enable_shared_from_this<Value> {
 public:
  double data;
  double grad;
  std::function<void()> _backward;
  std::unordered_set<ValuePtr> _prev;
  std::string _op;

  Value(double data, std::unordered_set<ValuePtr> children = {},
        std::string op = "")
      : data(data), grad(0.0), _backward([]() {}), _prev(children), _op(op) {};

  ~Value();

  //  Factory method for creating ValuePtr
  static ValuePtr make(double data, std::unordered_set<ValuePtr> children = {},
                       std::string op = "") {
    return std::make_shared<Value>(data, children, op);
  }

  ValuePtr operator+(const ValuePtr& other);
  ValuePtr operator*(const ValuePtr& other);

  ValuePtr pow(double exponent);

  ValuePtr relu();

  ValuePtr operator-();

  ValuePtr operator-(const ValuePtr& other);
  ValuePtr operator/(const ValuePtr& other);
  ValuePtr operator=(const ValuePtr& other);

  void backward();

  void print(std::ostream& out) const;
};

std::ostream& operator<<(std::ostream& os, const ValuePtr& val);

#endif