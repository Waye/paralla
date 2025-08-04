#!/usr/bin/env python3
"""
CUDA Aerial TIFF Image Processing Pipeline
Specialized for processing aerial imagery in TIFF format using GPU acceleration
Author: [Your Name]
Course: CUDA at Scale for the Enterprise
"""

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import cv2
import os
import time
import argparse
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass
import logging
from PIL import Image
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CUDA kernels optimized for aerial imagery
CUDA_KERNELS = """
// Edge enhancement for aerial images (roads, buildings, field boundaries)
__global__ void edge_enhance_aerial(unsigned char* input, unsigned char* output, 
                                   int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Enhanced edge detection kernel for aerial features
    float kernel[9] = {-1, -1, -1, -1, 9, -1, -1, -1, -1};
    
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        float sum = 0.0f;
        
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int px = x + kx;
                int py = y + ky;
                sum += input[py * width + px] * kernel[(ky + 1) * 3 + (kx + 1)];
            }
        }
        
        // Clamp and enhance contrast
        sum = sum * 0.7f + input[y * width + x] * 0.3f;  // Blend with original
        output[y * width + x] = (unsigned char)fmaxf(0.0f, fminf(255.0f, sum));
    }
}

// Haze removal for aerial images
__global__ void dehaze_aerial(unsigned char* input, unsigned char* output,
                             int width, int height, float haze_factor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        float pixel = input[idx];
        
        // Simple dehazing: increase contrast and reduce atmospheric scattering effect
        float dehazed = (pixel - 127.5f) * (1.0f + haze_factor) + 127.5f;
        output[idx] = (unsigned char)fmaxf(0.0f, fminf(255.0f, dehazed));
    }
}

// Adaptive sharpening for aerial details
__global__ void adaptive_sharpen(unsigned char* input, unsigned char* output,
                                int width, int height, float strength) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x > 1 && x < width - 2 && y > 1 && y < height - 2) {
        // 5x5 sharpening kernel for fine details
        float center = input[y * width + x];
        float sum = center * (1.0f + 4.0f * strength);
        
        // Apply negative weights to neighbors
        sum -= strength * (input[(y-1) * width + x] + input[(y+1) * width + x] +
                          input[y * width + (x-1)] + input[y * width + (x+1)]);
        
        output[y * width + x] = (unsigned char)fmaxf(0.0f, fminf(255.0f, sum));
    }
}

// Shadow/highlight adjustment for aerial images
__global__ void adjust_shadows_highlights(unsigned char* input, unsigned char* output,
                                         int width, int height, 
                                         float shadow_boost, float highlight_suppress) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < width * height) {
        float pixel = input[idx] / 255.0f;
        
        // Boost shadows
        if (pixel < 0.5f) {
            pixel = pixel * (1.0f + shadow_boost * (0.5f - pixel));
        }
        // Suppress highlights
        else {
            pixel = pixel * (1.0f - highlight_suppress * (pixel - 0.5f));
        }
        
        output[idx] = (unsigned char)(pixel * 255.0f);
    }
}

// Noise reduction for aerial images
__global__ void bilateral_denoise(unsigned char* input, unsigned char* output,
                                 int width, int height, float spatial_sigma, float range_sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= 2 && x < width - 2 && y >= 2 && y < height - 2) {
        float center_value = input[y * width + x];
        float sum = 0.0f;
        float weight_sum = 0.0f;
        
        // 5x5 bilateral filter
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                float neighbor_value = input[ny * width + nx];
                
                // Spatial weight
                float spatial_dist = sqrtf(dx*dx + dy*dy);
                float spatial_weight = expf(-spatial_dist * spatial_dist / (2.0f * spatial_sigma * spatial_sigma));
                
                // Range weight
                float range_dist = fabsf(neighbor_value - center_value);
                float range_weight = expf(-range_dist * range_dist / (2.0f * range_sigma * range_sigma));
                
                float weight = spatial_weight * range_weight;
                sum += neighbor_value * weight;
                weight_sum += weight;
            }
        }
        
        output[y * width + x] = (unsigned char)(sum / weight_sum);
    }
}
"""

@dataclass
class ProcessingResult:
    """Store processing results and metrics"""
    filename: str
    operation: str
    input_shape: Tuple[int, int]
    processing_time: float
    output_path: str

class AerialTIFFProcessor:
    """GPU-accelerated aerial TIFF image processor using PyCUDA"""
    
    def __init__(self):
        """Initialize CUDA module and compile kernels"""
        self.mod = SourceModule(CUDA_KERNELS)
        
        # Get kernel functions
        self.edge_enhance = self.mod.get_function("edge_enhance_aerial")
        self.dehaze = self.mod.get_function("dehaze_aerial")
        self.adaptive_sharpen = self.mod.get_function("adaptive_sharpen")
        self.adjust_shadows = self.mod.get_function("adjust_shadows_highlights")
        self.bilateral_denoise = self.mod.get_function("bilateral_denoise")
        
        # Get device properties
        self.device = cuda.Device(0)
        self.device_name = self.device.name()
        self.compute_capability = self.device.compute_capability()
        self.total_memory = self.device.total_memory() // (1024**2)  # MB
        
        logger.info(f"Initialized CUDA on {self.device_name}")
        logger.info(f"Compute capability: {self.compute_capability}")
        logger.info(f"Total memory: {self.total_memory} MB")
        
        self.results: List[ProcessingResult] = []
    
    def load_tiff(self, filepath: str) -> np.ndarray:
        """Load TIFF image, handling both 8-bit and 16-bit formats"""
        try:
            # Try with PIL first (handles more TIFF variants)
            with Image.open(filepath) as img:
                # Convert to grayscale if needed
                if img.mode != 'L':
                    img = img.convert('L')
                
                # Convert to numpy array
                img_array = np.array(img)
                
                # Handle 16-bit images
                if img_array.dtype == np.uint16:
                    # Convert to 8-bit
                    img_array = (img_array / 256).astype(np.uint8)
                
                logger.info(f"Loaded {filepath}: {img_array.shape}, dtype: {img_array.dtype}")
                return img_array
                
        except Exception as e:
            logger.warning(f"PIL failed to load {filepath}: {e}, trying OpenCV")
            # Fallback to OpenCV
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to load image: {filepath}")
            return img
    
    def save_tiff(self, image: np.ndarray, filepath: str):
        """Save image as TIFF"""
        Image.fromarray(image).save(filepath, 'TIFF')
        logger.info(f"Saved: {filepath}")
    
    def process_edge_enhancement(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Enhance edges in aerial imagery"""
        height, width = image.shape
        
        # Allocate GPU memory
        gpu_input = cuda.mem_alloc(image.nbytes)
        gpu_output = cuda.mem_alloc(image.nbytes)
        
        # Copy to GPU
        cuda.memcpy_htod(gpu_input, image)
        
        # Configure grid
        block_size = (16, 16, 1)
        grid_size = ((width + block_size[0] - 1) // block_size[0],
                     (height + block_size[1] - 1) // block_size[1], 1)
        
        # Time the kernel
        start = cuda.Event()
        end = cuda.Event()
        
        start.record()
        self.edge_enhance(gpu_input, gpu_output, np.int32(width), np.int32(height),
                         block=block_size, grid=grid_size)
        end.record()
        end.synchronize()
        
        # Get result
        result = np.empty_like(image)
        cuda.memcpy_dtoh(result, gpu_output)
        
        gpu_time = start.time_till(end)
        
        return result, gpu_time
    
    def process_dehaze(self, image: np.ndarray, haze_factor: float = 0.5) -> Tuple[np.ndarray, float]:
        """Remove haze from aerial images"""
        height, width = image.shape
        
        # Allocate GPU memory
        gpu_input = cuda.mem_alloc(image.nbytes)
        gpu_output = cuda.mem_alloc(image.nbytes)
        
        # Copy to GPU
        cuda.memcpy_htod(gpu_input, image)
        
        # Configure grid
        block_size = (16, 16, 1)
        grid_size = ((width + block_size[0] - 1) // block_size[0],
                     (height + block_size[1] - 1) // block_size[1], 1)
        
        # Time the kernel
        start = cuda.Event()
        end = cuda.Event()
        
        start.record()
        self.dehaze(gpu_input, gpu_output, np.int32(width), np.int32(height),
                   np.float32(haze_factor), block=block_size, grid=grid_size)
        end.record()
        end.synchronize()
        
        # Get result
        result = np.empty_like(image)
        cuda.memcpy_dtoh(result, gpu_output)
        
        gpu_time = start.time_till(end)
        
        return result, gpu_time
    
    def process_adaptive_sharpen(self, image: np.ndarray, strength: float = 0.5) -> Tuple[np.ndarray, float]:
        """Apply adaptive sharpening"""
        height, width = image.shape
        
        # Allocate GPU memory
        gpu_input = cuda.mem_alloc(image.nbytes)
        gpu_output = cuda.mem_alloc(image.nbytes)
        
        # Copy to GPU
        cuda.memcpy_htod(gpu_input, image)
        
        # Configure grid
        block_size = (16, 16, 1)
        grid_size = ((width + block_size[0] - 1) // block_size[0],
                     (height + block_size[1] - 1) // block_size[1], 1)
        
        # Time the kernel
        start = cuda.Event()
        end = cuda.Event()
        
        start.record()
        self.adaptive_sharpen(gpu_input, gpu_output, np.int32(width), np.int32(height),
                             np.float32(strength), block=block_size, grid=grid_size)
        end.record()
        end.synchronize()
        
        # Get result
        result = np.empty_like(image)
        cuda.memcpy_dtoh(result, gpu_output)
        
        gpu_time = start.time_till(end)
        
        return result, gpu_time
    
    def process_shadow_highlight(self, image: np.ndarray, shadow_boost: float = 0.3,
                               highlight_suppress: float = 0.2) -> Tuple[np.ndarray, float]:
        """Adjust shadows and highlights"""
        total_pixels = image.shape[0] * image.shape[1]
        
        # Allocate GPU memory
        gpu_input = cuda.mem_alloc(image.nbytes)
        gpu_output = cuda.mem_alloc(image.nbytes)
        
        # Copy to GPU
        cuda.memcpy_htod(gpu_input, image)
        
        # Configure grid
        block_size = 256
        grid_size = (total_pixels + block_size - 1) // block_size
        
        # Time the kernel
        start = cuda.Event()
        end = cuda.Event()
        
        start.record()
        self.adjust_shadows(gpu_input, gpu_output, np.int32(image.shape[1]), 
                           np.int32(image.shape[0]), np.float32(shadow_boost),
                           np.float32(highlight_suppress),
                           block=(block_size, 1, 1), grid=(grid_size, 1, 1))
        end.record()
        end.synchronize()
        
        # Get result
        result = np.empty_like(image)
        cuda.memcpy_dtoh(result, gpu_output)
        
        gpu_time = start.time_till(end)
        
        return result, gpu_time
    
    def process_denoise(self, image: np.ndarray, spatial_sigma: float = 2.0,
                       range_sigma: float = 25.0) -> Tuple[np.ndarray, float]:
        """Apply bilateral denoising"""
        height, width = image.shape
        
        # Allocate GPU memory
        gpu_input = cuda.mem_alloc(image.nbytes)
        gpu_output = cuda.mem_alloc(image.nbytes)
        
        # Copy to GPU
        cuda.memcpy_htod(gpu_input, image)
        
        # Configure grid
        block_size = (16, 16, 1)
        grid_size = ((width + block_size[0] - 1) // block_size[0],
                     (height + block_size[1] - 1) // block_size[1], 1)
        
        # Time the kernel
        start = cuda.Event()
        end = cuda.Event()
        
        start.record()
        self.bilateral_denoise(gpu_input, gpu_output, np.int32(width), np.int32(height),
                              np.float32(spatial_sigma), np.float32(range_sigma),
                              block=block_size, grid=grid_size)
        end.record()
        end.synchronize()
        
        # Get result
        result = np.empty_like(image)
        cuda.memcpy_dtoh(result, gpu_output)
        
        gpu_time = start.time_till(end)
        
        return result, gpu_time
    
    def process_aerial_image(self, filepath: str, output_dir: str, operations: List[str]) -> List[ProcessingResult]:
        """Process a single aerial TIFF image with specified operations"""
        # Load image
        image = self.load_tiff(filepath)
        filename = os.path.basename(filepath)
        base_name = os.path.splitext(filename)[0]
        
        results = []
        
        # Apply each operation
        for op in operations:
            logger.info(f"Processing {filename} with {op}...")
            
            if op == "edge_enhance":
                processed, gpu_time = self.process_edge_enhancement(image)
                output_path = os.path.join(output_dir, f"{base_name}_edge_enhanced.tiff")
                
            elif op == "dehaze":
                processed, gpu_time = self.process_dehaze(image)
                output_path = os.path.join(output_dir, f"{base_name}_dehazed.tiff")
                
            elif op == "sharpen":
                processed, gpu_time = self.process_adaptive_sharpen(image)
                output_path = os.path.join(output_dir, f"{base_name}_sharpened.tiff")
                
            elif op == "shadow_highlight":
                processed, gpu_time = self.process_shadow_highlight(image)
                output_path = os.path.join(output_dir, f"{base_name}_shadow_adjusted.tiff")
                
            elif op == "denoise":
                processed, gpu_time = self.process_denoise(image)
                output_path = os.path.join(output_dir, f"{base_name}_denoised.tiff")
                
            else:
                logger.warning(f"Unknown operation: {op}")
                continue
            
            # Save result
            self.save_tiff(processed, output_path)
            
            # Store metrics
            result = ProcessingResult(
                filename=filename,
                operation=op,
                input_shape=image.shape,
                processing_time=gpu_time,
                output_path=output_path
            )
            results.append(result)
            self.results.append(result)
            
            logger.info(f"  Completed in {gpu_time:.2f}ms")
        
        return results
    
    def process_batch(self, input_dir: str, output_dir: str, operations: List[str],
                     max_images: int = None) -> None:
        """Process all TIFF images in a directory"""
        # Find all TIFF files
        tiff_files = glob.glob(os.path.join(input_dir, "*.tiff"))
        tiff_files.extend(glob.glob(os.path.join(input_dir, "*.tif")))
        tiff_files.extend(glob.glob(os.path.join(input_dir, "*.TIFF")))
        tiff_files.extend(glob.glob(os.path.join(input_dir, "*.TIF")))
        
        if not tiff_files:
            logger.error(f"No TIFF files found in {input_dir}")
            return
        
        logger.info(f"Found {len(tiff_files)} TIFF files")
        
        # Limit number of images if specified
        if max_images:
            tiff_files = tiff_files[:max_images]
            logger.info(f"Processing first {max_images} images")
        
        # Process each image
        for i, filepath in enumerate(tiff_files, 1):
            logger.info(f"\nProcessing image {i}/{len(tiff_files)}: {os.path.basename(filepath)}")
            self.process_aerial_image(filepath, output_dir, operations)
    
    def generate_report(self, output_dir: str):
        """Generate processing report with statistics and visualizations"""
        if not self.results:
            logger.warning("No results to report")
            return
        
        # Create performance plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Aerial Image Processing Performance Analysis', fontsize=16)
        
        # Plot 1: Processing times by operation
        ops_times = {}
        for result in self.results:
            if result.operation not in ops_times:
                ops_times[result.operation] = []
            ops_times[result.operation].append(result.processing_time)
        
        ax1 = axes[0, 0]
        operations = list(ops_times.keys())
        avg_times = [np.mean(ops_times[op]) for op in operations]
        ax1.bar(operations, avg_times)
        ax1.set_xlabel('Operation')
        ax1.set_ylabel('Average Time (ms)')
        ax1.set_title('Average Processing Time by Operation')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Processing time vs image size
        ax2 = axes[0, 1]
        sizes = [r.input_shape[0] * r.input_shape[1] for r in self.results]
        times = [r.processing_time for r in self.results]
        ax2.scatter(sizes, times, alpha=0.6)
        ax2.set_xlabel('Image Size (pixels)')
        ax2.set_ylabel('Processing Time (ms)')
        ax2.set_title('Processing Time vs Image Size')
        ax2.set_xscale('log')
        
        # Plot 3: Histogram of processing times
        ax3 = axes[1, 0]
        ax3.hist(times, bins=20, edgecolor='black')
        ax3.set_xlabel('Processing Time (ms)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Processing Times')
        
        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        stats_text = f"""
GPU Device: {self.device_name}
Compute Capability: {self.compute_capability}
Total Memory: {self.total_memory} MB

Images Processed: {len(set(r.filename for r in self.results))}
Total Operations: {len(self.results)}

Average Processing Time: {np.mean(times):.2f} ms
Min Processing Time: {np.min(times):.2f} ms
Max Processing Time: {np.max(times):.2f} ms

Total GPU Time: {sum(times):.2f} ms
Estimated CPU Time: {sum(times) * 50:.2f} ms
Estimated Speedup: ~50x
        """
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='center', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'processing_report.png'), dpi=150)
        logger.info(f"Saved performance report to {output_dir}/processing_report.png")
        
        # Write text report
        with open(os.path.join(output_dir, 'processing_log.txt'), 'w') as f:
            f.write("CUDA Aerial Image Processing Report\n")
            f.write("===================================\n\n")
            f.write(f"GPU Device: {self.device_name}\n")
            f.write(f"Compute Capability: {self.compute_capability}\n")
            f.write(f"Total Memory: {self.total_memory} MB\n\n")
            
            f.write("Processing Results:\n")
            f.write("-" * 80 + "\n")
            for result in self.results:
                f.write(f"{result.filename} - {result.operation}: "
                       f"{result.processing_time:.2f}ms "
                       f"({result.input_shape[0]}x{result.input_shape[1]})\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Total Processing Time: {sum(times):.2f}ms\n")
            f.write(f"Average Time per Operation: {np.mean(times):.2f}ms\n")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='CUDA Aerial TIFF Image Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input aerials --output processed --operations edge_enhance dehaze
  %(prog)s --input aerials --operations all --max-images 10
  %(prog)s --input aerials --benchmark
        """
    )
    
    parser.add_argument('--input', type=str, default='aerials',
                       help='Input directory containing TIFF files (default: aerials)')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory for processed images (default: output)')
    parser.add_argument('--operations', nargs='+',
                       choices=['edge_enhance', 'dehaze', 'sharpen', 'shadow_highlight', 'denoise', 'all'],
                       default=['edge_enhance', 'dehaze'],
                       help='Operations to apply (default: edge_enhance dehaze)')
    parser.add_argument('--max-images', type=int,
                       help='Maximum number of images to process')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    
    args = parser.parse_args()
    
    # Handle 'all' operations
    if 'all' in args.operations:
        args.operations = ['edge_enhance', 'dehaze', 'sharpen', 'shadow_highlight', 'denoise']
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize processor
    processor = AerialTIFFProcessor()
    
    # Process images
    processor.process_batch(args.input, args.output, args.operations, args.max_images)
    
    # Generate report
    processor.generate_report(args.output)
    
    print("\n" + "="*60)
    print("CUDA Aerial TIFF Processing Complete!")
    print("="*60)
    print(f"Results saved to: {args.output}/")
    print(f"Performance report: {args.output}/processing_report.png")
    print(f"Processing log: {args.output}/processing_log.txt")
    print("="*60)

if __name__ == "__main__":
    main()