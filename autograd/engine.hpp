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
using Children = std::unordered_set<ValuePtr>;

class Value : public std::enable_shared_from_this<Value> {
 public:
  double data;
  double grad;
  std::function<void(double)> _backward;
  Children _prev;
  std::string _op;

  Value(double data, Children children = {}, std::string op = "")
      : data(data),
        grad(0.0),
        _backward([](double) {}),
        _prev(std::move(children)),
        _op(op) {};

  Value(const Value&) = delete;
  Value& operator=(const Value&) = delete;

  //  Factory method for creating ValuePtr
  static ValuePtr make(double data, Children children = {},
                       std::string op = "") {
    return std::make_shared<Value>(data, std::move(children), op);
  }

  ValuePtr pow(double exponent);

  ValuePtr relu();
  ValuePtr tanh();

  void backward();

  void print(std::ostream& out) const;
};

ValuePtr operator+(const ValuePtr& a, const ValuePtr& b);
ValuePtr operator+(const ValuePtr& a, double b);
ValuePtr operator+(double a, const ValuePtr& b);

ValuePtr operator*(const ValuePtr& a, const ValuePtr& b);
ValuePtr operator*(const ValuePtr& a, double b);
ValuePtr operator*(double a, const ValuePtr& b);

ValuePtr operator-(const ValuePtr& a, const ValuePtr& b);
ValuePtr operator-(const ValuePtr& a, double b);
ValuePtr operator-(double a, const ValuePtr& b);

ValuePtr operator/(const ValuePtr& a, const ValuePtr& b);
ValuePtr operator/(const ValuePtr& a, double b);
ValuePtr operator/(double a, const ValuePtr& b);
ValuePtr operator-(const ValuePtr& a);  // Negation

std::ostream& operator<<(std::ostream& os, const ValuePtr& val);

#endif