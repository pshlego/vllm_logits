# SPDX-License-Identifier: Apache-2.0

import enum
from typing import Union

from cutlass_library import *

#
#   Extend cutlass library with custom types, and missing values
#


class VLLM_LOGITSDataType(enum.Enum):
    u4b8 = enum_auto()
    u8b128 = enum_auto()


class MixedInputKernelScheduleType(enum.Enum):
    TmaWarpSpecialized = enum_auto()
    TmaWarpSpecializedPingpong = enum_auto()
    TmaWarpSpecializedCooperative = enum_auto()


VLLM_LOGITSDataTypeNames: dict[Union[VLLM_LOGITSDataType, DataType], str] = {
    **DataTypeNames,  # type: ignore
    **{
        VLLM_LOGITSDataType.u4b8: "u4b8",
        VLLM_LOGITSDataType.u8b128: "u8b128",
    }
}

VLLM_LOGITSDataTypeTag: dict[Union[VLLM_LOGITSDataType, DataType], str] = {
    **DataTypeTag,  # type: ignore
    **{
        VLLM_LOGITSDataType.u4b8: "cutlass::vllm_logits_uint4b8_t",
        VLLM_LOGITSDataType.u8b128: "cutlass::vllm_logits_uint8b128_t",
    }
}

VLLM_LOGITSDataTypeSize: dict[Union[VLLM_LOGITSDataType, DataType], int] = {
    **DataTypeSize,  # type: ignore
    **{
        VLLM_LOGITSDataType.u4b8: 4,
        VLLM_LOGITSDataType.u8b128: 8,
    }
}

VLLM_LOGITSDataTypeVLLM_LOGITSScalarTypeTag: dict[Union[VLLM_LOGITSDataType, DataType], str] = {
    VLLM_LOGITSDataType.u4b8: "vllm_logits::kU4B8",
    VLLM_LOGITSDataType.u8b128: "vllm_logits::kU8B128",
    DataType.u4: "vllm_logits::kU4",
    DataType.u8: "vllm_logits::kU8",
    DataType.s4: "vllm_logits::kS4",
    DataType.s8: "vllm_logits::kS8",
    DataType.f16: "vllm_logits::kFloat16",
    DataType.bf16: "vllm_logits::kBfloat16",
}

VLLM_LOGITSDataTypeTorchDataTypeTag: dict[Union[VLLM_LOGITSDataType, DataType], str] = {
    DataType.u8: "at::ScalarType::Byte",
    DataType.s8: "at::ScalarType::Char",
    DataType.e4m3: "at::ScalarType::Float8_e4m3fn",
    DataType.s32: "at::ScalarType::Int",
    DataType.f16: "at::ScalarType::Half",
    DataType.bf16: "at::ScalarType::BFloat16",
    DataType.f32: "at::ScalarType::Float",
}

VLLM_LOGITSKernelScheduleTag: dict[Union[
    MixedInputKernelScheduleType, KernelScheduleType], str] = {
        **KernelScheduleTag,  # type: ignore
        **{
            MixedInputKernelScheduleType.TmaWarpSpecialized:
            "cutlass::gemm::KernelTmaWarpSpecialized",
            MixedInputKernelScheduleType.TmaWarpSpecializedPingpong:
            "cutlass::gemm::KernelTmaWarpSpecializedPingpong",
            MixedInputKernelScheduleType.TmaWarpSpecializedCooperative:
            "cutlass::gemm::KernelTmaWarpSpecializedCooperative",
        }
    }
