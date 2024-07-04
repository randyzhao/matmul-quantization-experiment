load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary", "cc_test")


cc_library(
  name = "run_fp32_lib",
  srcs = ["run_fp32.cc"],
  hdrs = ["run_fp32.h"],
)

cc_binary(
  name = "run_fp32",
  srcs = ["run_fp32.cc"],
  deps = [":run_fp32_lib"]
)

cc_test(
  name = "run_fp32_test",
  srcs = ["run_fp32_test.cc"],
  deps = [
    ":run_fp32_lib",
    "@com_google_googletest//:gtest_main",
  ],
)
