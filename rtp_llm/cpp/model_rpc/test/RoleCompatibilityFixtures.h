#pragma once

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <google/protobuf/text_format.h>
#include <gtest/gtest.h>
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

inline void expectCompatibilityFixture(const std::string& name, const RoleCompatibilityPB& descriptor) {
    std::string text;
    ASSERT_TRUE(google::protobuf::TextFormat::PrintToString(descriptor, &text));
    const auto* outputs = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");
    if (outputs != nullptr) {
        std::ofstream(std::string(outputs) + "/" + name) << text;
    }
    const std::string path = std::string(std::getenv("TEST_SRCDIR")) + "/" + std::getenv("TEST_WORKSPACE")
                             + "/rtp_llm/cpp/model_rpc/test/testdata/compatibility/" + name;
    std::ifstream input(path);
    ASSERT_TRUE(input.good()) << path;
    std::string         expected_text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    RoleCompatibilityPB expected;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(expected_text, &expected));
    EXPECT_EQ(expected.SerializeAsString(), descriptor.SerializeAsString()) << text;
}

}  // namespace rtp_llm
