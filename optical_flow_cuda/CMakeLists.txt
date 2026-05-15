cmake_minimum_required(VERSION 3.18)
project(OpticalFlowCUDA LANGUAGES CXX CUDA)

# Yêu cầu CMake xuất file chứa đường dẫn cho VS Code đọc
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Nới lỏng kiểm tra của CMake và ép trực tiếp cờ C++17
set(CMAKE_CXX_STANDARD 17)

# Nới lỏng kiểm tra của CMake và ép trực tiếp cờ C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")

# Tối ưu hóa kiến trúc GPU
set(CMAKE_CUDA_ARCHITECTURES 86)

# Tìm các gói thư viện bắt buộc
find_package(OpenCV REQUIRED)

# Chỉ định thư mục chứa file header (.h, .cuh)
include_directories(
    ${CMAKE_SOURCE_DIR}/include 
    ${OpenCV_INCLUDE_DIRS}
)

# Gom tất cả các file mã nguồn
file(GLOB CUDA_SOURCES "cuda/*.cu")
file(GLOB CPP_SOURCES "src/*.cpp")

# Tạo file thực thi
add_executable(${PROJECT_NAME} ${CUDA_SOURCES} ${CPP_SOURCES})

# Liên kết thư viện
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} cudart)