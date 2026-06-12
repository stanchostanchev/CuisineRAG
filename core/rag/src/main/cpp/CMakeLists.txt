cmake_minimum_required(VERSION 3.22.1)
project(cuisine_rag_jni)

# ── llama.cpp headers ─────────────────────────────────────────────────────────
# Place the llama.cpp public headers here:
#   core/rag/src/main/cpp/include/llama.h
#   core/rag/src/main/cpp/include/ggml.h   (and other ggml headers)
#
# OR point LLAMA_INCLUDE_DIR at your local llama.cpp clone.
set(LLAMA_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# ── Prebuilt libllama.so ──────────────────────────────────────────────────────
# Place the prebuilt .so for each ABI under:
#   core/rag/src/main/jniLibs/arm64-v8a/libllama.so
#   core/rag/src/main/jniLibs/x86_64/libllama.so   (emulator)
#
# Obtain from: https://github.com/ggerganov/llama.cpp/releases
# Or build from source with the Android NDK (see README.md).
add_library(llama SHARED IMPORTED)
set_target_properties(llama PROPERTIES
    IMPORTED_LOCATION
    "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libllama.so"
)

# ── Our JNI bridge ────────────────────────────────────────────────────────────
add_library(
    cuisine_rag_jni
    SHARED
    llama_jni.cpp
)

target_include_directories(cuisine_rag_jni PRIVATE ${LLAMA_INCLUDE_DIR})

target_link_libraries(
    cuisine_rag_jni
    llama
    android
    log
)

# Export as libllama_jni.so — loaded by System.loadLibrary("llama_jni")
# Update LlamaCppEngine.kt companion if you rename:
#   System.loadLibrary("llama_jni")
