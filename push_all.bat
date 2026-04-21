Copy

@echo off
echo ================================================
echo   Step 1/4: Staging and committing changes...
echo ================================================
 
:: Check if commit message was provided
if "%~1"=="" (
    echo ERROR: Please provide a commit message
    echo Usage: push_all.bat "your message here"
    pause
    exit /b 1
)
 
git add .
git commit -m "%~1"
 
echo.
echo ================================================
echo   Step 2/4: Pushing to GitHub...
echo ================================================
git push origin main
 
echo.
echo ================================================
echo   Step 3/4: Pushing to Hugging Face Space...
echo ================================================
git push space main
 
echo.
echo ================================================
echo   Step 4a/4: Rebuilding Docker image locally...
echo ================================================
docker build -t wasp-193b/docmind:latest .
 
echo.
echo ================================================
echo   Step 4b/4: Pushing image to Docker Hub...
echo ================================================
docker push wasp-193b/docmind:latest
 
echo.
echo ================================================
echo   ALL DONE!
echo   - GitHub:        updated
echo   - HuggingFace:   updated
echo   - Docker Hub:    updated
echo   - Local Docker:  rebuilt
echo ================================================
pause
 