#include <cstdint>
#include <vector>
#include <algorithm>

#include "run_int8.h"

namespace INT8 {

QuantizedTensor quantize(std::vector<float>& data) {
  auto maxAbsElement = std::max_element(data.begin(), data.end(),
    [](float a, float b) {
      return std::abs(a) < std::abs(b);
    });
  float scaler = std::abs(*maxAbsElement);
  std::vector<int8_t> quantizedData(data.size());

  std::transform(data.begin(), data.end(), quantizedData.begin(),
    [scaler](float x) { return (int8_t)(x / scaler); });

  return QuantizedTensor { std::move(quantizedData), scaler };
}

void matmul(
  std::vector<float>& out,
  const QuantizedTensor& w,
  const QuantizedTensor& x,
  int d,
  int n
) {
  // W (d, n)
  // X (T, n) @ W^T (n, d) -> out (T, d)

  int T = x.data.size() / n;

  for (int t = 0; t < T; ++t) {
    for (int i = 0; i < d; ++i) {
      int32_t acc = 0;
      for (int j = 0; j < n; ++j) {
        acc += static_cast<int32_t>(x.data[t * n + j]) *
               static_cast<int32_t>(w.data[d * n + j]);
      }
      out[t * d + i] = static_cast<float>(acc) * x.scaler * w.scaler;
    }
  }
}

void relu(std::vector<float>& out, const QuantizedTensor& in) {
  std::transform(in.data.begin(), in.data.end(), out.begin(),
    [&in](int8_t n) { return static_cast<float>(std::max((int8_t)0, n)) * in.scaler; });
}

void forward(std::vector<float>& out, Weights& weights, std::vector<float>& in, int n) {
  int T = in.size() / n;

  QuantizedTensor qin = quantize(in),
    qw1 = quantize(weights.w1),
    qw2 = quantize(weights.w2),
    qw3 = quantize(weights.w3);

  std::vector<float> logits(T * n);
  std::vector<float> activation(T * n);

  // three layers of matmul + relu

  matmul(logits, qw1, qin, n, n);
  relu(activation, quantize(logits));

  matmul(logits, qw2, quantize(activation), n, n);
  relu(activation, quantize(logits));

  matmul(logits, qw3, quantize(activation), n, n);
  relu(out, quantize(logits));
}

}
