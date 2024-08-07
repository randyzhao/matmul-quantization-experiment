#include <vector>
#include <random>
#include <random>
#include <ctime>
#include <iostream>

#include "common.h"
#include "run_fp32.h"
#include "run_int8.h"

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

void outputTopLeft(std::vector<float> data, int outputSize, int n) {
  for (int i = 0; i < outputSize; ++i) {
    for (int j = 0; j < outputSize; ++j) {
      std::cout << data[i * n + j] << " ";
    }
    std::cout << std::endl;
  }
}

int main() {
  int T = 128;
  int n = 32;

  std::vector<float> in = randomGenerateFP32(T * n);
  Weights wei;
  wei.w1 = randomGenerateFP32(n * n, 0.1);
  wei.w2 = randomGenerateFP32(n * n, 0.3);
  wei.w3 = randomGenerateFP32(n * n, 0.5);

  std::vector<float> outFP32(T * n), outINT8(T * n);

  FP32::forward(outFP32, wei, in, n);
  INT8::forward(outINT8, wei, in, n);

  std::cout << "fp32 result: " << std::endl;
  outputTopLeft(outFP32, 10, n);

  std::cout << "int8 result: " << std::endl;
  outputTopLeft(outINT8, 10, n);
}
