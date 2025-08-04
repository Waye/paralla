@echo off
echo ==========================================
echo CUDA Aerial TIFF Processing - Examples
echo ==========================================

REM Example 1: Basic edge enhancement and dehaze
echo.
echo Example 1: Basic processing (edge enhancement + dehaze)
echo Command: python aerial_tiff_processor.py --input aerials --operations edge_enhance dehaze
python aerial_tiff_processor.py --input aerials --output output/example1 --operations edge_enhance dehaze --max-images 5

REM Example 2: All operations
echo.
echo ==========================================
echo Example 2: All operations
echo Command: python aerial_tiff_processor.py --operations all
python aerial_tiff_processor.py --input aerials --output output/example2 --operations all --max-images 3

REM Example 3: Specific operations for aerial clarity
echo.
echo ==========================================
echo Example 3: Aerial clarity enhancement
echo Command: python aerial_tiff_processor.py --operations dehaze sharpen shadow_highlight
python aerial_tiff_processor.py --input aerials --output output/example3 --operations dehaze sharpen shadow_highlight --max-images 5

REM Example 4: Noise reduction and sharpening
echo.
echo ==========================================
echo Example 4: Denoise and sharpen
echo Command: python aerial_tiff_processor.py --operations denoise sharpen edge_enhance
python aerial_tiff_processor.py --input aerials --output output/example4 --operations denoise sharpen edge_enhance --max-images 5

echo.
echo ==========================================
echo All examples complete!
echo Results saved in output/example1, example2, example3, example4
echo ==========================================
pause