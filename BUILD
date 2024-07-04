load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary", "cc_test")


cc_library(
  name = "float_attention_lib",
  srcs = ["float_attention.cc"],
  hdrs = ["float_attention.hpp"],
)

cc_binary(
  name = "float_attention",
  srcs = ["float_attention"],
)

cc_test(
  name = "float_attention_test",
  srcs = ["float_attention_test.cc"],
  deps = [
    ":float_attention_lib",
    "@com_google_googletest//:gtest_main",
  ],
)
