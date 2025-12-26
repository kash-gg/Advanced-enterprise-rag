# PowerShell Setup Script for Advanced Multi-Source RAG
# Run this script to set up the project environment on Windows

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Advanced Multi-Source RAG - Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "[1/7] Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found. Please install Python 3.9-3.12." -ForegroundColor Red
    exit 1
}

# Extract version number
$versionMatch = $pythonVersion -match "Python (\d+)\.(\d+)"
if ($versionMatch) {
    $majorVersion = [int]$Matches[1]
    $minorVersion = [int]$Matches[2]
    
    # Check if version is in supported range (3.9 - 3.12)
    if ($majorVersion -ne 3 -or $minorVersion -lt 9 -or $minorVersion -gt 12) {
        Write-Host "ERROR: Python $majorVersion.$minorVersion is not supported!" -ForegroundColor Red
        Write-Host "This project requires Python 3.9, 3.10, 3.11, or 3.12" -ForegroundColor Red
        Write-Host "Reason: spaCy (required dependency) doesn't support Python 3.13+ yet" -ForegroundColor Yellow
        Write-Host "" -ForegroundColor Yellow
        Write-Host "Please install Python 3.12 from: https://www.python.org/downloads/" -ForegroundColor Cyan
        exit 1
    }
}

Write-Host "Found: $pythonVersion ✓" -ForegroundColor Green

# Create virtual environment
Write-Host ""
Write-Host "[2/7] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists. Skipping..." -ForegroundColor Yellow
}
else {
    python -m venv venv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Virtual environment created successfully!" -ForegroundColor Green
    }
    else {
        Write-Host "ERROR: Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
}

# Activate virtual environment
Write-Host ""
Write-Host "[3/7] Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
if ($LASTEXITCODE -eq 0) {
    Write-Host "Virtual environment activated!" -ForegroundColor Green
}
else {
    Write-Host "ERROR: Failed to activate virtual environment." -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host ""
Write-Host "[4/7] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
Write-Host "Pip upgraded!" -ForegroundColor Green

# Install dependencies
Write-Host ""
Write-Host "[5/7] Installing dependencies..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "Dependencies installed successfully!" -ForegroundColor Green
}
else {
    Write-Host "ERROR: Failed to install dependencies." -ForegroundColor Red
    exit 1
}

# Download spaCy model
Write-Host ""
Write-Host "[6/7] Downloading spaCy NER model..." -ForegroundColor Yellow
python -m spacy download en_core_web_sm
if ($LASTEXITCODE -eq 0) {
    Write-Host "spaCy model downloaded!" -ForegroundColor Green
}
else {
    Write-Host "WARNING: Failed to download spaCy model. You can download it later." -ForegroundColor Yellow
}

# Create directory structure
Write-Host ""
Write-Host "[7/7] Creating project directories..." -ForegroundColor Yellow
$directories = @(
    "data",
    "data/documents",
    "data/cache",
    "data/faiss_index",
    "models",
    "logs",
    "tests",
    "tests/sample_data",
    "docs",
    "src/ingestion",
    "src/processing",
    "src/indexing",
    "src/retrieval",
    "src/reranking",
    "src/llm",
    "src/ui"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created: $dir" -ForegroundColor Green
    }
    else {
        Write-Host "Exists: $dir" -ForegroundColor Yellow
    }
}

# Create .env from .env.example if it doesn't exist
Write-Host ""
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env file from .env.example" -ForegroundColor Green
    Write-Host "Please review and update .env with your configuration." -ForegroundColor Yellow
}
else {
    Write-Host ".env file already exists." -ForegroundColor Yellow
}

# Create __init__.py files in subdirectories
$initFiles = @(
    "src/ingestion/__init__.py",
    "src/processing/__init__.py",
    "src/indexing/__init__.py",
    "src/retrieval/__init__.py",
    "src/reranking/__init__.py",
    "src/llm/__init__.py",
    "src/ui/__init__.py"
)

Write-Host ""
Write-Host "Creating package __init__.py files..." -ForegroundColor Yellow
foreach ($file in $initFiles) {
    if (-not (Test-Path $file)) {
        New-Item -ItemType File -Path $file -Force | Out-Null
        Write-Host "Created: $file" -ForegroundColor Green
    }
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Review and update .env file with your configuration" -ForegroundColor White
Write-Host "2. Activate the virtual environment: .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "3. Test the configuration: python -c 'from src.config import config; print(config)'" -ForegroundColor White
Write-Host ""
Write-Host "To run the application (once implemented):" -ForegroundColor Yellow
Write-Host "  streamlit run src/app.py" -ForegroundColor White
Write-Host ""
