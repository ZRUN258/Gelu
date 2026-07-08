file(GLOB host_src ./*.cpp)
npu_op_code_gen(
    SRC ${host_src}
    PACKAGE ${package_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
)
npu_op_library(cust_optiling TILING ${host_src})
file(GLOB autogen_aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
set_source_files_properties(${autogen_aclnn_src} PROPERTIES GENERATED TRUE)
npu_op_library(cust_opapi ACLNN
    ${autogen_aclnn_src}
)

npu_op_package_add(${package_name}
    LIBRARY
        cust_optiling
        cust_opapi
)