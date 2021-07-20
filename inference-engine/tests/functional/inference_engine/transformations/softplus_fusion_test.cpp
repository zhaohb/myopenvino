// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/softplus_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, SoftPlusFusing) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input0 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto exp = std::make_shared<ngraph::opset4::Exp>(input0);
        auto input_const = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1.0});
        auto add = std::make_shared<ngraph::opset4::Add>(exp, input_const);
        auto log = std::make_shared<ngraph::opset4::Log>(add);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{log}, ngraph::ParameterVector{input0});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::SoftPlusFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto softplus = std::make_shared<ngraph::opset4::SoftPlus>(data);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{softplus}, ngraph::ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SoftPlusFusingDynamic) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input0 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(1));
        auto exp = std::make_shared<ngraph::opset4::Exp>(input0);
        auto input_const = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1.0});
        auto add = std::make_shared<ngraph::opset4::Add>(exp, input_const);
        auto log = std::make_shared<ngraph::opset4::Log>(add);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{log}, ngraph::ParameterVector{input0});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::SoftPlusFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(1));
        auto softplus = std::make_shared<ngraph::opset4::SoftPlus>(data);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{softplus}, ngraph::ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SoftPlusFusingNegative) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input0 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(1));
        auto exp = std::make_shared<ngraph::opset4::Exp>(input0);
        auto input_const = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {-1.0});
        auto add = std::make_shared<ngraph::opset4::Add>(exp, input_const);
        auto log = std::make_shared<ngraph::opset4::Log>(add);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{log}, ngraph::ParameterVector{input0});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::SoftPlusFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input0 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(1));
        auto exp = std::make_shared<ngraph::opset4::Exp>(input0);
        auto input_const = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {-1.0});
        auto add = std::make_shared<ngraph::opset4::Add>(exp, input_const);
        auto log = std::make_shared<ngraph::opset4::Log>(add);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{log}, ngraph::ParameterVector{input0});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
