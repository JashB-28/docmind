@echo off
:: Always run from the repo root, no matter where this script lives
cd /d "%~dp0.."

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
docker build -t jash09/docmind:latest .

echo.
echo ================================================
echo   Step 4b/4: Pushing image to Docker Hub...
echo ================================================
docker push jash09/docmind:latest

echo.
echo ================================================
echo   ALL DONE!
echo   - GitHub:        updated (JashB-28/docmind)
echo   - HuggingFace:   updated (WASP-193b/docmind)
echo   - Docker Hub:    updated (jash09/docmind)
echo   - Local Docker:  rebuilt
echo ================================================
pause