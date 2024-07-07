#include "run_fp32.h"
#include <gtest/gtest.h>
#include <vector>

#include <iostream>

// Test case for matmul function
TEST(MatmulTest, BasicTest) {
  std::vector<float> w = {
    1, 6, 0, 7,
    6, 4, 2, 8,
    4, 0, 3, 5
  };
  std::vector<float> x = {
    1, 3, 4, 0,
    6, 1, 5, 9
  };
  int d = 3;
  int n = 4;
  std::vector<float> out(d * (x.size() / n), 0);

  FP32::matmul(out, w, x, d, n);

  std::vector<float> expected = {
    19, 26, 16,
    75, 122, 84
  };

  ASSERT_EQ(out.size(), expected.size());
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_FLOAT_EQ(out[i], expected[i]);
  }
}
