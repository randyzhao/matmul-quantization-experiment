#include <vector>

#include "run_fp32.h"

namespace FP32 {


void matmul(
  std::vector<float>& out,
  std::vector<float>& w,
  std::vector<float>& x,
  int d,
  int n
) {
  // W (d, n)
  // X (T, n) @ W^T (n, d) -> out (T, d)
  int T = x.size() / n;
  for (int t = 0; t < T; ++t) {
    for (int i = 0; i < d; ++i) {
      float result = 0;
      for (int j = 0; j < n; ++j) {
        result += w[i * n + j] * x[i];
      }
      out[t * d + i] = result;
    }
  }
}

void relu(std::vector<float>& out, std::vector<float>& in) {
  for (int i = 0; i < in.size(); ++i) {
    out[i] = in[i] > 0 ? in[i] : 0;
  }
}

void forward(ForwardState& state, Weights& weights, std::vector<float>& in, int d, int n) {
  for (int i = 0; i < 3; ++i) {
    matmul(state.logits, weights.w1, i == 0 ? in : state.activation, d, n);
    relu(state.activation, state.logits);
  }
}

}

int main() {
  return 0;
}
