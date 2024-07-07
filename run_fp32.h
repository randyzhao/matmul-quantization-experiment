#pragma once

#include <vector>

#include "common.h"

namespace FP32 {

struct ForwardState {
  std::vector<float> logits;
  std::vector<float> activation;
};

void matmul(
  std::vector<float>& out,
  std::vector<float>& w,
  std::vector<float>& x,
  int d,
  int n
);

void relu(std::vector<float>& out, std::vector<float>& in);

void forward(std::vector<float>& out, Weights& weights, std::vector<float>& in, int n);

}
