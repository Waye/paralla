// cuda_image_processor.cu
// Simple CUDA image processing without external dependencies

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <cmath>

// Simple image structure
struct Image {
    int width;
    int height;
    std::vector<unsigned char> data;
    
    Image() : width(0), height(0) {}
    Image(int w, int h) : width(w), height(h), data(w * h) {}
};

// Error checking
#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

// Simple 3x3 convolution kernel
__global__ void convolution3x3(unsigned char* input, unsigned char* output, 
                               int width, int height, float* kernel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        float sum = 0.0f;
        
        // Apply 3x3 kernel
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int px = x + kx;
                int py = y + ky;
                int idx = py * width + px;
                int kidx = (ky + 1) * 3 + (kx + 1);
                sum += input[idx] * kernel[kidx];
            }
        }
        
        // Clamp to valid range
        sum = fmaxf(0.0f, fminf(255.0f, sum));
        output[y * width + x] = (unsigned char)sum;
    }
}

// Edge detection kernel (Sobel X)
__global__ void sobelX(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        float gx = -1 * input[(y-1)*width + (x-1)] + 1 * input[(y-1)*width + (x+1)] +
                   -2 * input[y*width + (x-1)]     + 2 * input[y*width + (x+1)] +
                   -1 * input[(y+1)*width + (x-1)] + 1 * input[(y+1)*width + (x+1)];
        
        float magnitude = fabsf(gx);
        output[y * width + x] = (unsigned char)fminf(255.0f, magnitude);
    }
}

// Brightness adjustment kernel
__global__ void adjustBrightness(unsigned char* input, unsigned char* output, 
                                int size, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float value = input[idx] * factor;
        output[idx] = (unsigned char)fmaxf(0.0f, fminf(255.0f, value));
    }
}

// Simple PGM image I/O (Portable GrayMap - no external libs needed)
bool loadPGM(const std::string& filename, Image& img) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;
    
    std::string magic;
    file >> magic;
    if (magic != "P5") return false;
    
    file >> img.width >> img.height;
    int maxval;
    file >> maxval;
    file.get(); // consume newline
    
    img.data.resize(img.width * img.height);
    file.read(reinterpret_cast<char*>(img.data.data()), img.data.size());
    
    return file.good();
}

bool savePGM(const std::string& filename, const Image& img) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;
    
    file << "P5\n" << img.width << " " << img.height << "\n255\n";
    file.write(reinterpret_cast<const char*>(img.data.data()), img.data.size());
    
    return file.good();
}

// Create test image
void createTestImage(Image& img, const std::string& pattern) {
    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            int idx = y * img.width + x;
            
            if (pattern == "gradient") {
                img.data[idx] = (unsigned char)((x + y) * 255 / (img.width + img.height));
            }
            else if (pattern == "checkerboard") {
                img.data[idx] = ((x/50) + (y/50)) % 2 ? 255 : 0;
            }
            else if (pattern == "circle") {
                int cx = img.width / 2, cy = img.height / 2;
                float dist = sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy));
                img.data[idx] = dist < 200 ? 255 : 50;
            }
        }
    }
}

// Process image on GPU
void processImageGPU(const Image& input, Image& output, const std::string& operation) {
    size_t imageSize = input.width * input.height * sizeof(unsigned char);
    
    // Allocate device memory
    unsigned char *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, imageSize));
    CHECK_CUDA(cudaMalloc(&d_output, imageSize));
    
    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, input.data.data(), imageSize, cudaMemcpyHostToDevice));
    
    // Set up execution configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((input.width + blockSize.x - 1) / blockSize.x,
                  (input.height + blockSize.y - 1) / blockSize.y);
    
    // Timer
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    
    // Execute operation
    if (operation == "sharpen") {
        float h_kernel[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
        float* d_kernel;
        CHECK_CUDA(cudaMalloc(&d_kernel, 9 * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel, 9 * sizeof(float), cudaMemcpyHostToDevice));
        
        convolution3x3<<<gridSize, blockSize>>>(d_input, d_output, input.width, input.height, d_kernel);
        
        CHECK_CUDA(cudaFree(d_kernel));
    }
    else if (operation == "blur") {
        float h_kernel[9] = {1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f, 1/9.0f};
        float* d_kernel;
        CHECK_CUDA(cudaMalloc(&d_kernel, 9 * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel, 9 * sizeof(float), cudaMemcpyHostToDevice));
        
        convolution3x3<<<gridSize, blockSize>>>(d_input, d_output, input.width, input.height, d_kernel);
        
        CHECK_CUDA(cudaFree(d_kernel));
    }
    else if (operation == "edge") {
        sobelX<<<gridSize, blockSize>>>(d_input, d_output, input.width, input.height);
    }
    else if (operation == "brighten") {
        int totalPixels = input.width * input.height;
        int blockSize1D = 256;
        int gridSize1D = (totalPixels + blockSize1D - 1) / blockSize1D;
        adjustBrightness<<<gridSize1D, blockSize1D>>>(d_input, d_output, totalPixels, 1.5f);
    }
    
    // Record time
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    
    std::cout << "  GPU " << operation << ": " << milliseconds << " ms" << std::endl;
    
    // Copy result back
    CHECK_CUDA(cudaMemcpy(output.data.data(), d_output, imageSize, cudaMemcpyDeviceToHost));
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main(int argc, char** argv) {
    std::cout << "==================================" << std::endl;
    std::cout << "CUDA Image Processing Demo" << std::endl;
    std::cout << "==================================" << std::endl;
    
    // Get GPU info
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total Memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB" << std::endl;
    std::cout << std::endl;
    
    // Create test images
    std::vector<std::pair<std::string, std::pair<int, int>>> testCases = {
        {"gradient", {512, 512}},
        {"checkerboard", {1024, 1024}},
        {"circle", {2048, 2048}}
    };
    
    // Ensure output directory exists
    system("mkdir output 2>nul");
    
    // Process each test image
    for (const auto& test : testCases) {
        std::string pattern = test.first;
        int width = test.second.first;
        int height = test.second.second;
        
        std::cout << "Processing " << pattern << " (" << width << "x" << height << ")" << std::endl;
        
        // Create test image
        Image input(width, height);
        createTestImage(input, pattern);
        savePGM("output/test_" + pattern + "_original.pgm", input);
        
        // Process with different operations
        std::vector<std::string> operations = {"sharpen", "blur", "edge", "brighten"};
        
        for (const auto& op : operations) {
            Image output(width, height);
            processImageGPU(input, output, op);
            savePGM("output/test_" + pattern + "_" + op + ".pgm", output);
        }
        
        std::cout << std::endl;
    }
    
    // Write summary
    std::ofstream log("output/processing_log.txt");
    log << "CUDA Image Processing Results\n";
    log << "=============================\n";
    log << "GPU: " << prop.name << "\n";
    log << "Images processed: " << testCases.size() << "\n";
    log << "Operations: sharpen, blur, edge detection, brightness\n";
    log << "\nOutput files are in PGM format.\n";
    log << "View with: IrfanView, GIMP, ImageMagick, or any PGM viewer.\n";
    log.close();
    
    std::cout << "Processing complete!" << std::endl;
    std::cout << "Results saved in 'output' directory" << std::endl;
    std::cout << "\nNote: Output files are in PGM format." << std::endl;
    std::cout << "To convert to JPG: Use IrfanView, GIMP, or ImageMagick" << std::endl;
    
    return 0;
}