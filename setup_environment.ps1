# AUJ Platform Environment Setup Script for PowerShell
# ====================================================
# This script sets up the AUJ Platform environment and runs basic validation.
# 
# Author: AUJ Platform Development Team
# Date: 2025-07-04
# Version: 1.0.0

param(
    [switch]$Verbose,
    [string]$PlatformRoot = $PSScriptRoot
)

# Set error action preference
$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AUJ Platform Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host

# Resolve platform root path
$PlatformRoot = Resolve-Path $PlatformRoot
Write-Host "Platform Root: $PlatformRoot" -ForegroundColor Green
Write-Host

# Set environment variables
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONUNBUFFERED = "1"
$env:AUJ_PLATFORM_ROOT = $PlatformRoot
$env:AUJ_SRC_PATH = Join-Path $PlatformRoot "auj_platform\src"
$env:AUJ_CONFIG_PATH = Join-Path $PlatformRoot "config"
$env:AUJ_LOGS_PATH = Join-Path $PlatformRoot "logs"

Write-Host "Environment variables set:" -ForegroundColor Yellow
Write-Host "  AUJ_PLATFORM_ROOT = $env:AUJ_PLATFORM_ROOT"
Write-Host "  AUJ_SRC_PATH = $env:AUJ_SRC_PATH"
Write-Host "  AUJ_CONFIG_PATH = $env:AUJ_CONFIG_PATH"
Write-Host "  AUJ_LOGS_PATH = $env:AUJ_LOGS_PATH"
Write-Host

# Check if Python is available
$pythonCmd = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $version = & $cmd --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            $pythonCmd = $cmd
            Write-Host "Found Python command: $cmd" -ForegroundColor Green
            Write-Host "Python version: $version"
            break
        }
    }
    catch {
        # Continue to next command
    }
}

if (-not $pythonCmd) {
    Write-Host "ERROR: Python is not available in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ and ensure it's in your PATH" -ForegroundColor Red
    exit 1
}

Write-Host

# Check Python version (must be 3.8+)
try {
    $versionOutput = & $pythonCmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    $versionParts = $versionOutput.Split('.')
    $majorVersion = [int]$versionParts[0]
    $minorVersion = [int]$versionParts[1]
    
    if ($majorVersion -lt 3 -or ($majorVersion -eq 3 -and $minorVersion -lt 8)) {
        Write-Host "ERROR: Python 3.8+ is required, but found Python $versionOutput" -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "WARNING: Could not determine Python version" -ForegroundColor Yellow
}

# Check if required directories exist and create them if needed
Write-Host "Checking directory structure..." -ForegroundColor Yellow

$requiredDirs = @(
    $env:AUJ_SRC_PATH,
    $env:AUJ_CONFIG_PATH,
    $env:AUJ_LOGS_PATH,
    (Join-Path $env:AUJ_SRC_PATH "core")
)

foreach ($dir in $requiredDirs) {
    if (-not (Test-Path $dir)) {
        Write-Host "Creating missing directory: $dir" -ForegroundColor Yellow
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

Write-Host "Directory structure verified." -ForegroundColor Green
Write-Host

# Run Python environment setup if available
Write-Host "Running Python environment setup..." -ForegroundColor Yellow
$setupScript = Join-Path $env:AUJ_SRC_PATH "core\environment_setup.py"

if (Test-Path $setupScript) {
    Push-Location $PlatformRoot
    try {
        $setupArgs = @($setupScript)
        if ($Verbose) {
            $setupArgs += "--verbose"
        }
        
        & $pythonCmd $setupArgs
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Python environment setup completed successfully" -ForegroundColor Green
        }
        else {
            Write-Host "WARNING: Python environment setup reported issues" -ForegroundColor Yellow
            Write-Host "Please check the log files for details" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "ERROR: Failed to run Python environment setup: $_" -ForegroundColor Red
    }
    finally {
        Pop-Location
    }
}
else {
    Write-Host "WARNING: environment_setup.py not found" -ForegroundColor Yellow
    Write-Host "Skipping Python environment setup" -ForegroundColor Yellow
}

Write-Host

# Run path cleanup analysis if available
Write-Host "Running path cleanup analysis..." -ForegroundColor Yellow
$cleanupScript = Join-Path $env:AUJ_SRC_PATH "core\path_cleanup.py"

if (Test-Path $cleanupScript) {
    Push-Location $PlatformRoot
    try {
        $cleanupArgs = @(
            $cleanupScript,
            "--platform-root", $PlatformRoot,
            "--dry-run"
        )
        if ($Verbose) {
            $cleanupArgs += "--verbose"
        }
        
        & $pythonCmd $cleanupArgs
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Path cleanup analysis completed successfully" -ForegroundColor Green
        }
        else {
            Write-Host "WARNING: Path cleanup analysis reported issues" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "ERROR: Failed to run path cleanup analysis: $_" -ForegroundColor Red
    }
    finally {
        Pop-Location
    }
}
else {
    Write-Host "WARNING: path_cleanup.py not found" -ForegroundColor Yellow
    Write-Host "Skipping path cleanup analysis" -ForegroundColor Yellow
}

Write-Host

# Basic validation
Write-Host "Running basic validation..." -ForegroundColor Yellow

# Check for core files
$coreFiles = @(
    (Join-Path $env:AUJ_SRC_PATH "core\__init__.py"),
    (Join-Path $env:AUJ_SRC_PATH "core\config.py")
)

$missingFiles = @()
foreach ($file in $coreFiles) {
    if (-not (Test-Path $file)) {
        $missingFiles += (Split-Path $file -Leaf)
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host "WARNING: Missing core files: $($missingFiles -join ', ')" -ForegroundColor Yellow
    Write-Host "Platform may not function correctly" -ForegroundColor Yellow
}
else {
    Write-Host "Core files validation passed" -ForegroundColor Green
}

# Test basic Python import
Write-Host "Testing basic Python imports..." -ForegroundColor Yellow
Push-Location $PlatformRoot
try {
    $importTest = "import sys; sys.path.insert(0, r'$env:AUJ_SRC_PATH'); print('Python path setup: OK')"
    & $pythonCmd -c $importTest 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Python import test passed" -ForegroundColor Green
    }
    else {
        Write-Host "WARNING: Python import test failed" -ForegroundColor Yellow
        Write-Host "There may be issues with the Python path setup" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "WARNING: Python import test failed with exception: $_" -ForegroundColor Yellow
}
finally {
    Pop-Location
}

Write-Host
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Environment Setup Complete" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host
Write-Host "The AUJ Platform environment has been configured." -ForegroundColor Green
Write-Host
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Verify all required dependencies are installed"
Write-Host "  2. Review configuration files in: $env:AUJ_CONFIG_PATH"
Write-Host "  3. Check logs in: $env:AUJ_LOGS_PATH"
Write-Host "  4. Test platform functionality"
Write-Host
Write-Host "To run the platform:" -ForegroundColor Yellow
Write-Host "  cd `"$PlatformRoot`""
Write-Host "  $pythonCmd -m auj_platform.src.main"
Write-Host

# Set execution policy reminder
Write-Host "Note: If you encounter execution policy errors, run:" -ForegroundColor Cyan
Write-Host "  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Cyan
Write-Host

exit 0