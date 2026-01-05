@echo off
cd /d "%~dp0"

:: 1. Kich hoat moi truong ao (neu co)
if exist .venv\Scripts\activate (
    echo Dang kich hoat .venv...
    call .venv\Scripts\activate
)

:: 2. Dam bao remote dung
git remote remove origin 2>nul
git remote add origin https://github.com/bathanh0309/ISIC_2018.git

:: 3. Thuc hien quy trinh Git
echo Dang them file vao git...
git add .

echo Su dung tieu de: project NeuralTrans
git commit -m "project NeuralTrans"

echo Dang dong bo voi remote...
git branch -M main
git pull --rebase origin main
if errorlevel 1 (
    echo Pull rebase that bai, dang abort...
    git rebase --abort
    echo Dang force push...
    git push --force origin main
) else (
    echo Dang day code len GitHub...
    git push origin main
)

echo Hoan tat!
pause