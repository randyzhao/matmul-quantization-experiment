#include "float_attention.hpp"
#include <vector>

struct ForwardState {
  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> val;
  std::vector<float> attention;
  std::vector<float> logits;
};

struct Weights {
  std::vector<float> q1, q2;
  std::vector<float> k1, k2;
  std::vector<float> v1, v2;

  std::vector<float> att_l1, att_l2;

  std::vector<float> ffd_l1, ffd_l2;
};

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

void forward(ForwardState& state, Weights& weights, std::vector<float>& in, int d, int n) {
  matmul(state.key, weights.k1, in, d, n);
  matmul(state.query, weights.q1, in, d, n);
  matmul(state.val, weights.v1, in, d, n);
}

int main() {
  return 0;
}
