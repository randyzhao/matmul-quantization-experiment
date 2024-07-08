#include <gtest/gtest.h>
#include <vector>
#include <iostream>

#include "run_int8.h"

TEST(QuantizeTest, BasicTest) {
  std::vector<float> fp32 = {-10, 2, 3, 4, 5, 8};
  INT8::QuantizedTensor int8 = INT8::quantize(fp32);

  std::vector<int8_t> expectedQuantized = {-127, 25, 38, 50, 63, 101};
  EXPECT_EQ(int8.data, expectedQuantized);

  float expectedScaler = 0.078740157;
  EXPECT_FLOAT_EQ(int8.scaler, expectedScaler);
}


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

  INT8::QuantizedTensor qw = INT8::quantize(w);
  INT8::QuantizedTensor qx = INT8::quantize(x);
  INT8::matmul(out, qw, qx, d, n);

  std::vector<float> expected = {
    19, 26, 16,
    75, 122, 84
  };

  ASSERT_EQ(out.size(), expected.size());
  for (size_t i = 0; i < out.size(); ++i) {
    float ratio = out[i] / expected[i];
    EXPECT_TRUE(ratio >= 0.9 && ratio <= 1.1);
  }
}
