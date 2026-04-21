Copy

@echo off
echo ================================
echo   Pushing to All 3 + Docker
echo ================================
 
:: Check if commit message was provided
if "%~1"=="" (
    echo ERROR: Please provide a commit message
    echo Usage: push_all.bat "your message here"
    pause
    exit /b 1
)
 
echo.
echo Staging and committing changes...
git add .
git commit -m "%~1"
 
echo.
echo Pushing to GitHub...
git push origin main
 
echo.
echo Pushing to Hugging Face Space...
git push space main
 
echo.
echo Rebuilding Docker image...
docker build -t wasp-193b/docmind:latest .
 
echo.
echo Pushing Docker image to Docker Hub...
docker push wasp-193b/docmind:latest
 
echo.
echo ================================
echo   Done! All 3 remotes updated.
echo ================================
pause