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
