#pragma once

#include <vector>
#include <cstdint>

#include "common.h"

namespace INT8 {

struct QuantizedTensor {
  std::vector<int8_t> data;
  float scaler;
};

QuantizedTensor quantize(std::vector<float>& data);

void matmul(
  std::vector<float>& out,
  const QuantizedTensor& w,
  const QuantizedTensor& x,
  int d,
  int n
);

void relu(std::vector<float>& out, const QuantizedTensor& in);

void forward(std::vector<float>& out, Weights& weights, std::vector<float>& in, int n);

}
