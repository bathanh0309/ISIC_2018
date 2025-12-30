@echo off
echo ========================================================
echo AUTO DEPLOY SCRIPT FOR ISIC_2018
echo ========================================================

:: 1. Initialize Git if not already done
if not exist .git (
    echo [INFO] Initializing Git repository...
    git init
)

:: 2. Check if remote origin exists, if not add it
git remote get-url origin >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Adding remote origin...
    git remote add origin https://github.com/bathanh0309/ISIC_2018
) else (
    echo [INFO] Remote origin already exists.
)

:: Ensure we are on 'main' branch
echo [INFO] Renaming current branch to 'main'...
git branch -M main

:: 3. Add all project files
echo.
echo [STEP 1] Staging all project files...
git add .


echo [STEP 1] Committing all project files...
git commit -m "Add project NeuralTrans"

echo [STEP 1] Pulling latest changes from GitHub...
git pull origin main --rebase
if %errorlevel% neq 0 (
    echo [WARNING] Pull failed or conflicts detected. Attempting to continue...
)

echo [STEP 1] Pushing data folders to GitHub...
:: Using -u origin main to set upstream if first push
git push -u origin main
if %errorlevel% neq 0 (
    echo [ERROR] Failed to push to GitHub. Please check your credentials and internet connection.
    pause
    exit /b %errorlevel%
)

echo.
echo ========================================================
echo DEPLOYMENT COMPLETE!
echo ========================================================
pause
