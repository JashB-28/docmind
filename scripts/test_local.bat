@echo off
:: Always run from the repo root, no matter where this script lives
cd /d "%~dp0.."

echo ================================================
echo   Docker Local Test
echo ================================================
 
echo.
echo Stopping any running docmind containers...
for /f "tokens=1" %%i in ('docker ps -q --filter "ancestor=jash09/docmind:latest" 2^>nul') do docker stop %%i
 
echo.
echo Rebuilding Docker image...
docker build -t jash09/docmind:latest .
 
echo.
echo Starting container...
docker run -d -p 7860:7860 jash09/docmind:latest
 
echo.
echo ================================================
echo   Done! Open http://localhost:7860 to test
echo   Or click 7860:7860 in Docker Desktop
echo ================================================
pause