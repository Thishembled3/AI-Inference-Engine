cmake_minimum_required(VERSION 3.10)
project(AIInferenceEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/inference_engine.cpp
)

# Add executable
add_executable(inference_app main.cpp ${SOURCES})

# Link libraries (e.g., ONNX Runtime, if used)
# find_package(ONNXRuntime REQUIRED)
# target_link_libraries(inference_app PRIVATE ONNXRuntime::onnxruntime)

# Optional: Install target
install(TARGETS inference_app DESTINATION bin)
install(FILES include/inference_engine.h DESTINATION include)
// Update on 2023-05-01 00:00:00 - 260
// Update on 2023-08-03 00:00:00 - 290
