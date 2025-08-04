@echo off
echo ==========================================
echo CUDA Aerial TIFF Processing Setup
echo ==========================================

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python 3.7+
    pause
    exit /b 1
)

REM Install dependencies
echo.
echo Installing Python dependencies...
pip install pycuda numpy opencv-python matplotlib pillow

REM Check CUDA
echo.
echo Checking CUDA installation...
python -c "import pycuda.driver as cuda; cuda.init(); print(f'CUDA Devices: {cuda.Device.count()}')" 2>nul
if %errorlevel% neq 0 (
    echo ERROR: CUDA/PyCUDA not working properly!
    echo Please ensure:
    echo 1. NVIDIA drivers are installed
    echo 2. CUDA Toolkit is installed
    echo 3. PyCUDA is properly configured
    pause
    exit /b 1
)

REM Create output directory
if not exist output mkdir output

REM Check for aerial images
if not exist aerials (
    echo.
    echo WARNING: 'aerials' directory not found!
    echo Please ensure your TIFF files are in: %cd%\aerials\
    echo.
)

echo.
echo Setup complete! Running aerial processor...
echo ==========================================
echo.

REM Run with aerial images
python aerial_tiff_processor.py --input aerials --output output --operations edge_enhance dehaze sharpen

echo.
echo ==========================================
echo Processing complete!
echo Check the 'output' folder for results.
echo ==========================================
pause