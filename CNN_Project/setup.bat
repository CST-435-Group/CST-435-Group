@echo off
echo ========================================
echo Anime Image Classifier - Quick Setup
echo ========================================
echo.

echo [Step 1/3] Installing dependencies...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install kagglehub pandas numpy pillow opencv-python tqdm matplotlib seaborn streamlit requests scikit-learn

echo.
echo [Step 2/3] Checking Kaggle credentials...
if exist "%USERPROFILE%\.kaggle\kaggle.json" (
    echo ✅ Kaggle API credentials found
) else (
    echo ❌ Kaggle credentials NOT found
    echo.
    echo Please:
    echo 1. Go to https://www.kaggle.com/settings
    echo 2. Click "Create New API Token"
    echo 3. Save kaggle.json to: %USERPROFILE%\.kaggle\
    echo.
    pause
)

echo.
echo [Step 3/3] Checking GPU...
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Train model: python train_model.py
echo 2. Run app: streamlit run streamlit_app.py
echo.
pause
