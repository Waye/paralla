@echo off
echo Building CUDA Image Processor...

REM Create output directory
if not exist output mkdir output

REM Simple CUDA compilation - no external dependencies
nvcc -O3 cuda_image_processor.cu -o cuda_image_processor.exe

if %errorlevel%==0 (
    echo Build successful!
    echo.
    echo Running...
    cuda_image_processor.exe
    echo.
    echo Done! Check the 'output' folder for results.
) else (
    echo Build failed!
)

pause