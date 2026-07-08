// TilingKey模板定义的头文件。
//
// 同一个GELU算子要支持float16和float32。为了让Kernel里少做运行时if判断，
// Ascend C可以在编译期生成多个模板实例：
// - 输入是float16时，选择KernelGelu<half>；
// - 输入是float32时，选择KernelGelu<float>。
//
// Host侧在TilingFunc里调用ASCENDC_TPL_SEL_PARAM(context, dtype, use_x5, use_std_tanh)，
// 把实际dtype和两个shape特化开关传给框架；框架根据这里声明的规则选择对应Kernel版本。
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

// 声明Gelu算子的模板参数DT_INPUT_X，它来自第0个输入tensor的dtype。
// USE_X5_PATH把默认rational路径和x5路径拆成不同编译实例。
// USE_STD_TANH_PATH只给float32长循环形态使用标准tanh近似，避免污染默认短shape布局。
// 这样非x5形态不需要携带x5分支代码，减少运行时分支和代码布局互相干扰。
ASCENDC_TPL_ARGS_DECL(Gelu,
    ASCENDC_TPL_DATATYPE_DECL(DT_INPUT_X, C_DT_FLOAT16, C_DT_FLOAT, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_BOOL_DECL(USE_X5_PATH, 0, 1),
    ASCENDC_TPL_BOOL_DECL(USE_STD_TANH_PATH, 0, 1)
);

// 声明本算子实际支持的模板组合。
// 这里有八个组合：float16/float32各自再分默认路径、x5路径和长循环标准tanh路径。
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_BOOL_SEL(USE_X5_PATH, 0, 1),
        ASCENDC_TPL_BOOL_SEL(USE_STD_TANH_PATH, 0, 1)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT),
        ASCENDC_TPL_BOOL_SEL(USE_X5_PATH, 0, 1),
        ASCENDC_TPL_BOOL_SEL(USE_STD_TANH_PATH, 0, 1)
    ),
);
