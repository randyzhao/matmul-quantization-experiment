load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary", "cc_test")

cc_library(
  name = "common",
  hdrs = ["common.h"],
)

cc_library(
  name = "run_fp32_lib",
  srcs = ["run_fp32.cc"],
  hdrs = ["run_fp32.h"],
  deps = [
    ":common",
  ],
)

cc_library(
  name = "run_int8_lib",
  srcs = ["run_int8.cc"],
  hdrs = ["run_int8.h"],
  deps = [
    ":common",
  ],
)

cc_binary(
  name = "experiment",
  srcs = ["experiment.cc"],
  deps = [
    ":run_fp32_lib",
    ":run_int8_lib",
    ":common",
  ],
)

cc_test(
  name = "run_fp32_test",
  srcs = ["run_fp32_test.cc"],
  deps = [
    ":run_fp32_lib",
    "@com_google_googletest//:gtest_main",
  ],
)

cc_test(
  name = "run_int8_test",
  srcs = ["run_int8_test.cc"],
  deps = [
    ":run_int8_lib",
    "@com_google_googletest//:gtest_main",
  ]
)
