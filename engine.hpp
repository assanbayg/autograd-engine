#ifndef VALUE_
#define VALUE_

#include <cmath>
#include <functional>
#include <set>
#include <string>

class Value {
 public:
  double data;
  double grad;
  std::function<void()> _backward;
  std::set<Value*> _prev;
  std::string _op;

  Value(double data, std::set<Value*> children = {}, std::string op = "")
      : data(data), grad(0.0), _backward([]() {}), _prev(children), _op(op) {};

  ~Value();

  Value operator+(const Value& other) const;
  Value operator*(const Value& other) const;

  Value pow(double exponent) const {
    Value out(std::pow(this->data, exponent));
    return out;
  }

  Value relu() const;
  
  Value operator-();

  Value operator-(const Value& other) const;
  Value operator/(const Value& other) const;
  Value operator=(const Value& other) const;

  void backward();

  void print(std::ostream& out) const;
};

#endif