# vLLM flash attention requires VLLM_LOGITS_GPU_ARCHES to contain the set of target
# arches in the CMake syntax (75-real, 89-virtual, etc), since we clear the
# arches in the CUDA case (and instead set the gencodes on a per file basis)
# we need to manually set VLLM_LOGITS_GPU_ARCHES here.
if(VLLM_LOGITS_GPU_LANG STREQUAL "CUDA")
  foreach(_ARCH ${CUDA_ARCHS})
    string(REPLACE "." "" _ARCH "${_ARCH}")
    list(APPEND VLLM_LOGITS_GPU_ARCHES "${_ARCH}-real")
  endforeach()
endif()

#
# Build vLLM flash attention from source
#
# IMPORTANT: This has to be the last thing we do, because vllm_logits-flash-attn uses the same macros/functions as vLLM.
# Because functions all belong to the global scope, vllm_logits-flash-attn's functions overwrite vLLMs.
# They should be identical but if they aren't, this is a massive footgun.
#
# The vllm_logits-flash-attn install rules are nested under vllm_logits to make sure the library gets installed in the correct place.
# To only install vllm_logits-flash-attn, use --component _vllm_logits_fa2_C (for FA2) or --component _vllm_logits_fa3_C (for FA3).
# If no component is specified, vllm_logits-flash-attn is still installed.

# If VLLM_LOGITS_FLASH_ATTN_SRC_DIR is set, vllm_logits-flash-attn is installed from that directory instead of downloading.
# This is to enable local development of vllm_logits-flash-attn within vLLM.
# It can be set as an environment variable or passed as a cmake argument.
# The environment variable takes precedence.
if (DEFINED ENV{VLLM_LOGITS_FLASH_ATTN_SRC_DIR})
  set(VLLM_LOGITS_FLASH_ATTN_SRC_DIR $ENV{VLLM_LOGITS_FLASH_ATTN_SRC_DIR})
endif()

if(VLLM_LOGITS_FLASH_ATTN_SRC_DIR)
  FetchContent_Declare(
          vllm_logits-flash-attn SOURCE_DIR 
          ${VLLM_LOGITS_FLASH_ATTN_SRC_DIR}
          BINARY_DIR ${CMAKE_BINARY_DIR}/vllm_logits-flash-attn
  )
else()
  FetchContent_Declare(
          vllm_logits-flash-attn
          GIT_REPOSITORY https://github.com/vllm_logits-project/flash-attention.git
          GIT_TAG 8798f27777fb57f447070301bf33a9f9c607f491
          GIT_PROGRESS TRUE
          # Don't share the vllm_logits-flash-attn build between build types
          BINARY_DIR ${CMAKE_BINARY_DIR}/vllm_logits-flash-attn
  )
endif()


# Ensure the vllm_logits/vllm_logits_flash_attn directory exists before installation
install(CODE "file(MAKE_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/vllm_logits/vllm_logits_flash_attn\")" ALL_COMPONENTS)

# Make sure vllm_logits-flash-attn install rules are nested under vllm_logits/
# This is here to support installing all components under the same prefix with cmake --install.
# setup.py installs every component separately but uses the same prefix for all.
# ALL_COMPONENTS is used to avoid duplication for FA2 and FA3,
# and these statements don't hurt when installing neither component.
install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY FALSE)" ALL_COMPONENTS)
install(CODE "set(OLD_CMAKE_INSTALL_PREFIX \"\${CMAKE_INSTALL_PREFIX}\")" ALL_COMPONENTS)
install(CODE "set(CMAKE_INSTALL_PREFIX \"\${CMAKE_INSTALL_PREFIX}/vllm_logits/\")" ALL_COMPONENTS)

# Fetch the vllm_logits-flash-attn library
FetchContent_MakeAvailable(vllm_logits-flash-attn)
message(STATUS "vllm_logits-flash-attn is available at ${vllm_logits-flash-attn_SOURCE_DIR}")

# Restore the install prefix
install(CODE "set(CMAKE_INSTALL_PREFIX \"\${OLD_CMAKE_INSTALL_PREFIX}\")" ALL_COMPONENTS)
install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" ALL_COMPONENTS)

# Copy over the vllm_logits-flash-attn python files (duplicated for fa2 and fa3, in
# case only one is built, in the case both are built redundant work is done)
install(
  DIRECTORY ${vllm_logits-flash-attn_SOURCE_DIR}/vllm_logits_flash_attn/
  DESTINATION vllm_logits/vllm_logits_flash_attn
  COMPONENT _vllm_logits_fa2_C
  FILES_MATCHING PATTERN "*.py"
)

install(
  DIRECTORY ${vllm_logits-flash-attn_SOURCE_DIR}/vllm_logits_flash_attn/
  DESTINATION vllm_logits/vllm_logits_flash_attn
  COMPONENT _vllm_logits_fa3_C
  FILES_MATCHING PATTERN "*.py"
)
