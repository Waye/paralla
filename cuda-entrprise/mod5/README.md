# CUDA Image Processing Project

## Overview
This is a simple CUDA-based image processing application that demonstrates GPU acceleration for image filters. It requires NO external dependencies - just CUDA.

## Features
- **No OpenCV needed** - uses simple PGM image format
- **Multiple filters**: Sharpen, Blur, Edge Detection, Brightness
- **Automatic test image generation**
- **Performance timing for each operation**
- **Processes multiple image sizes** (512x512, 1024x1024, 2048x2048)

## Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- Windows/Linux/Mac

## Build and Run

### Windows:
```cmd
build.bat
```

### Linux/Mac:
```bash
nvcc -O3 cuda_image_processor.cu -o cuda_image_processor
./cuda_image_processor
```

## Output
The program creates test images and processes them with different filters:
- `test_gradient_original.pgm` - Original gradient test image
- `test_gradient_sharpen.pgm` - Sharpened version
- `test_gradient_blur.pgm` - Blurred version
- `test_gradient_edge.pgm` - Edge detection
- `test_gradient_brighten.pgm` - Brightened version
- ... (same for checkerboard and circle patterns)

## Viewing PGM Files
PGM files can be viewed with:
- **Windows**: IrfanView (free), Paint.NET, GIMP
- **Linux**: GIMP, ImageMagick (`display` command)
- **Online**: Many online PGM viewers
- **Convert to JPG**: `convert input.pgm output.jpg` (ImageMagick)

## Performance
The program shows GPU processing time for each operation. Typical results:
- 512x512: ~0.1-0.3 ms
- 1024x1024: ~0.3-0.8 ms  
- 2048x2048: ~1-3 ms

## Project Structure
```
mod5/
├── cuda_image_processor.cu  # Main CUDA source
├── build.bat               # Windows build script
├── README.md              # This file
└── output/                # Generated images (created automatically)
    ├── test_*.pgm        # Output images
    └── processing_log.txt # Processing summary
```

## For Course Submission
1. Run `build.bat`
2. Take screenshot of console output showing GPU info and timings
3. Include some `.pgm` files from output folder
4. Include `processing_log.txt`
5. Upload all to your repository

## Extending the Code
To add new filters, create a new CUDA kernel and add it to the operations list in main().

## Troubleshooting
- **"No CUDA devices found"**: Ensure NVIDIA drivers are installed
- **Compilation errors**: Check CUDA toolkit is in PATH
- **Can't view PGM files**: Use the converters mentioned above