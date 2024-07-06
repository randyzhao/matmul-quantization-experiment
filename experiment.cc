#include <vector>
#include <random>
#include <random>
#include <ctime>

#include "run_fp32.h"

std::vector<float> randomGenerateFP32(int size, float stddev=0.5) {
  // Initialize the random number generator with a random seed
  std::mt19937 rng(static_cast<unsigned int>(std::time(0)));
  // Create a normal distribution with the specified mean and standard deviation
  std::normal_distribution<float> distribution(0, stddev);


  std::vector<float> out(size, 0);

  // Fill the vector with random floats
  for (size_t i = 0; i < size; ++i) {
      out[i] = distribution(rng);
  }

  return out;
}

int main() {
  int T = 128;
  int n = 32;

  std::vector<float> in = randomGenerateFP32(T * n);
  FP32::Weights wei;
  wei.w1 = randomGenerateFP32(n * n);
  wei.w2 = randomGenerateFP32(n * n);

  std::vector<float> out(T * n);

  FP32::forward(out, wei, in, n);
}
