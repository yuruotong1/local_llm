@echo off
setlocal
cd /d "%~dp0"

set "APP_NAME=QwenChatUI"
set "DIST_DIR=%CD%\dist\%APP_NAME%"
set "BUILD_DIR=%CD%\build"
set "WORK_DIR=%CD%\build_work"
set "TORCH_EXTENSIONS_DIR=%CD%\.torch_extensions"
set "BUILD_MODE=%~1"
if /I "%BUILD_MODE%"=="" set "BUILD_MODE=fast"

if not exist "%TORCH_EXTENSIONS_DIR%" mkdir "%TORCH_EXTENSIONS_DIR%"

echo Build mode: %BUILD_MODE%

if /I "%BUILD_MODE%"=="full" (
  echo [1/5] Installing build tools and dependencies...
  python -m pip install pyinstaller
  if errorlevel 1 goto :error
  python -m pip install -r requirements.txt
  if errorlevel 1 goto :error

  echo [2/5] Cleaning old build output...
  if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
  if exist "%WORK_DIR%" rmdir /s /q "%WORK_DIR%"
) else (
  echo [1/5] Skipping dependency installation in fast mode...
  echo [2/5] Reusing build cache in fast mode...
)

if exist "%DIST_DIR%" rmdir /s /q "%DIST_DIR%"

echo [3/5] Ensuring model exists (will auto-download if missing)...
python download_model.py
if errorlevel 1 goto :error

echo [4/5] Building executable...
python -m PyInstaller ^
  --noconfirm ^
  --onedir ^
  --name "%APP_NAME%" ^
  --distpath "%CD%\dist" ^
  --workpath "%WORK_DIR%" ^
  --specpath "%WORK_DIR%" ^
  --collect-all streamlit ^
  --hidden-import=modelscope ^
  --hidden-import=torch ^
  --hidden-import=transformers ^
  launcher.py
if errorlevel 1 goto :error

echo [5/5] Copying app files and models...
xcopy "app" "%DIST_DIR%\app\" /E /I /Y >nul
if errorlevel 1 goto :error
xcopy "ui" "%DIST_DIR%\ui\" /E /I /Y >nul
if errorlevel 1 goto :error
copy /Y "download_model.py" "%DIST_DIR%\" >nul
if errorlevel 1 goto :error
xcopy "models" "%DIST_DIR%\models\" /E /I /Y >nul
if errorlevel 1 goto :error
if exist ".streamlit" (
  xcopy ".streamlit" "%DIST_DIR%\.streamlit\" /E /I /Y >nul
  if errorlevel 1 goto :error
)

echo.
echo Build complete.
echo Output folder: "%DIST_DIR%"
echo Run this file: "%DIST_DIR%\%APP_NAME%.exe"
echo Full rebuild: build_package.bat full
echo Fast rebuild: build_package.bat
goto :end

:error
echo.
echo Build failed.
exit /b 1

:end
endlocal
