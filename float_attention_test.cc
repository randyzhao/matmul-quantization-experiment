#include "float_attention.hpp"
#include <gtest/gtest.h>
#include <vector>

// Test case for matmul function
TEST(MatmulTest, BasicTest) {
  std::vector<float> w = {
    1, 2,
    3, 4,
    5, 6
  };
  std::vector<float> x = {
    1, 2,
    3, 4
  };
  int d = 3;
  int n = 2;
  std::vector<float> out(d * (x.size() / n), 0);

  matmul(out, w, x, d, n);

  std::vector<float> expected = {
    5, 11, 17,
    11, 25, 39
  };

  ASSERT_EQ(out.size(), expected.size());
  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_FLOAT_EQ(out[i], expected[i]);
  }
}
