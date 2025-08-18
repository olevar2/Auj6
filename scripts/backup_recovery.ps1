# AUJ Platform Backup and Recovery Script (PowerShell)
# Early Decision System Removal - Risk Mitigation Implementation

param(
    [Parameter(Position=0)]
    [ValidateSet("backup", "rollback", "status")]
    [string]$Action = "backup",
    
    [Parameter(Position=1)]
    [string]$BackupDate
)

# Configuration
$BackupDir = ".\backups"
$DateStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ProjectName = "auj_platform"

# Logging functions
function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] WARNING: $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] ERROR: $Message" -ForegroundColor Red
}

# Create backup directory
function New-BackupDirectory {
    $backupPath = Join-Path $BackupDir $DateStamp
    Write-Log "Creating backup directory: $backupPath"
    
    try {
        New-Item -Path $backupPath -ItemType Directory -Force | Out-Null
        Write-Log "Backup directory created successfully"
        return $true
    }
    catch {
        Write-Error "Failed to create backup directory: $_"
        return $false
    }
}

# Git repository backup
function Backup-Repository {
    Write-Log "Creating git repository backup..."
    
    try {
        # Create backup branch
        $branchName = "backup/before-early-decision-removal-$DateStamp"
        git checkout -b $branchName
        if ($LASTEXITCODE -eq 0) {
            Write-Log "Backup branch created: $branchName"
        } else {
            throw "Failed to create backup branch"
        }
        
        # Create archive of current state
        $archivePath = Join-Path $BackupDir $DateStamp "auj_platform_source_$DateStamp.tar.gz"
        git archive --format=tar.gz --output=$archivePath HEAD
        if ($LASTEXITCODE -eq 0) {
            Write-Log "Source code archive created successfully"
        } else {
            throw "Failed to create source code archive"
        }
        
        # Return to master branch
        git checkout master
        
        return $true
    }
    catch {
        Write-Error "Repository backup failed: $_"
        return $false
    }
}

# Database backup
function Backup-Databases {
    Write-Log "Creating database backups..."
    
    try {
        $dbFiles = Get-ChildItem -Path . -Filter "*.db" -Recurse -File
        
        foreach ($dbFile in $dbFiles) {
            $dbName = [System.IO.Path]::GetFileNameWithoutExtension($dbFile.Name)
            $backupPath = Join-Path $BackupDir $DateStamp "${dbName}_backup_$DateStamp.db"
            
            Write-Log "Backing up database: $($dbFile.FullName) -> $backupPath"
            Copy-Item -Path $dbFile.FullName -Destination $backupPath
            
            if (Test-Path $backupPath) {
                Write-Log "Database backup successful: $dbName"
            } else {
                throw "Database backup failed: $dbName"
            }
        }
        
        return $true
    }
    catch {
        Write-Error "Database backup failed: $_"
        return $false
    }
}

# Configuration backup
function Backup-Configurations {
    Write-Log "Creating configuration backups..."
    
    try {
        # Backup config directory
        $configBackup = Join-Path $BackupDir $DateStamp "config_backup_$DateStamp"
        Copy-Item -Path "config" -Destination $configBackup -Recurse
        
        if (Test-Path $configBackup) {
            Write-Log "Configuration backup successful"
        } else {
            throw "Configuration backup failed"
        }
        
        # Backup additional configuration files
        $configFiles = @("pyproject.toml", "setup_environment.ps1", "setup_environment.bat", "docker-compose*.yml")
        
        foreach ($pattern in $configFiles) {
            $files = Get-ChildItem -Path . -Filter $pattern -File
            foreach ($file in $files) {
                $destPath = Join-Path $BackupDir $DateStamp $file.Name
                Copy-Item -Path $file.FullName -Destination $destPath
                Write-Log "Backed up: $($file.Name)"
            }
        }
        
        return $true
    }
    catch {
        Write-Error "Configuration backup failed: $_"
        return $false
    }
}

# Performance baseline backup
function Backup-PerformanceBaseline {
    Write-Log "Creating performance baseline backup..."
    
    try {
        $reportPath = Join-Path $BackupDir $DateStamp "performance_baseline_$DateStamp.txt"
        
        $report = @"
AUJ Platform Performance Baseline Report
========================================
Date: $(Get-Date)
Commit: $(git rev-parse HEAD)
Branch: $(git branch --show-current)

System Information:
- OS: $($env:OS)
- Computer: $($env:COMPUTERNAME)
- PowerShell Version: $($PSVersionTable.PSVersion)

Database Files:
$(Get-ChildItem -Path . -Filter "*.db" -Recurse -File | Format-Table Name, Length, LastWriteTime | Out-String)

Git Status:
$(git status --porcelain | Out-String)

Last 5 Commits:
$(git log --oneline -5 | Out-String)
"@

        $report | Out-File -FilePath $reportPath -Encoding UTF8
        Write-Log "Performance baseline report created: $reportPath"
        return $true
    }
    catch {
        Write-Error "Performance baseline backup failed: $_"
        return $false
    }
}

# Validation of backup integrity
function Test-BackupIntegrity {
    Write-Log "Validating backup integrity..."
    
    $backupPath = Join-Path $BackupDir $DateStamp
    
    # Check if all expected files exist
    $requiredFiles = @(
        "auj_platform_source_$DateStamp.tar.gz",
        "config_backup_$DateStamp",
        "performance_baseline_$DateStamp.txt"
    )
    
    $allValid = $true
    foreach ($file in $requiredFiles) {
        $filePath = Join-Path $backupPath $file
        if (Test-Path $filePath) {
            Write-Log "✓ Backup file exists: $file"
        } else {
            Write-Error "✗ Missing backup file: $file"
            $allValid = $false
        }
    }
    
    if ($allValid) {
        Write-Log "All backup validations passed"
        return $true
    } else {
        Write-Error "Backup validation failed"
        return $false
    }
}

# Main backup function
function Invoke-CompleteBackup {
    Write-Log "Starting comprehensive backup procedure..."
    
    if (-not (New-BackupDirectory)) { return $false }
    if (-not (Backup-Repository)) { return $false }
    if (-not (Backup-Databases)) { return $false }
    if (-not (Backup-Configurations)) { return $false }
    if (-not (Backup-PerformanceBaseline)) { return $false }
    if (-not (Test-BackupIntegrity)) { return $false }
    
    Write-Log "Comprehensive backup completed successfully!"
    Write-Log "Backup location: $(Join-Path $BackupDir $DateStamp)"
    
    return $true
}

# Rollback function
function Invoke-Rollback {
    param([string]$BackupDate)
    
    if ([string]::IsNullOrEmpty($BackupDate)) {
        Write-Error "Backup date required for rollback. Usage: .\backup_recovery.ps1 rollback YYYYMMDD_HHMMSS"
        return $false
    }
    
    $backupPath = Join-Path $BackupDir $BackupDate
    
    if (-not (Test-Path $backupPath)) {
        Write-Error "Backup directory not found: $backupPath"
        return $false
    }
    
    Write-Warning "PERFORMING SYSTEM ROLLBACK - This will overwrite current state!"
    $confirmation = Read-Host "Are you sure? (yes/no)"
    
    if ($confirmation -ne "yes") {
        Write-Log "Rollback cancelled by user"
        return $false
    }
    
    Write-Log "Starting rollback procedure..."
    
    try {
        # Stop any running services (add specific stop commands here)
        Write-Warning "Stop AUJ Platform services before continuing"
        
        # Restore configuration
        Write-Log "Restoring configuration..."
        if (Test-Path "config") {
            Remove-Item -Path "config" -Recurse -Force
        }
        $configSource = Join-Path $backupPath "config_backup_$BackupDate"
        Copy-Item -Path $configSource -Destination "config" -Recurse
        
        # Restore databases
        Write-Log "Restoring databases..."
        $backupDbs = Get-ChildItem -Path $backupPath -Filter "*.db" -File
        foreach ($backupDb in $backupDbs) {
            $originalName = $backupDb.Name -replace "_backup_$BackupDate", ""
            Copy-Item -Path $backupDb.FullName -Destination $originalName
            Write-Log "Restored database: $originalName"
        }
        
        # Restore source code (if needed)
        Write-Log "Restoring source code from git backup branch..."
        git checkout "backup/before-early-decision-removal-$BackupDate"
        
        Write-Log "Rollback completed successfully!"
        Write-Warning "Please restart AUJ Platform services and validate system operation"
        
        return $true
    }
    catch {
        Write-Error "Rollback failed: $_"
        return $false
    }
}

# Display backup status
function Show-BackupStatus {
    Write-Log "Available backups:"
    
    if (Test-Path $BackupDir) {
        $backups = Get-ChildItem -Path $BackupDir -Directory
        foreach ($backup in $backups) {
            $size = (Get-ChildItem -Path $backup.FullName -Recurse | Measure-Object -Property Length -Sum).Sum
            $sizeStr = if ($size -gt 1GB) { "{0:N2} GB" -f ($size / 1GB) }
                      elseif ($size -gt 1MB) { "{0:N2} MB" -f ($size / 1MB) }
                      else { "{0:N2} KB" -f ($size / 1KB) }
            Write-Log "  $($backup.Name) ($sizeStr)"
        }
    } else {
        Write-Warning "No backup directory found"
    }
}

# Main script logic
switch ($Action) {
    "backup" {
        $result = Invoke-CompleteBackup
        exit $(if ($result) { 0 } else { 1 })
    }
    "rollback" {
        $result = Invoke-Rollback -BackupDate $BackupDate
        exit $(if ($result) { 0 } else { 1 })
    }
    "status" {
        Show-BackupStatus
        exit 0
    }
    default {
        Write-Host "Usage: .\backup_recovery.ps1 {backup|rollback|status}"
        Write-Host "  backup          - Perform comprehensive backup"
        Write-Host "  rollback DATE   - Rollback to specific backup (YYYYMMDD_HHMMSS)"
        Write-Host "  status          - Show available backups"
        exit 1
    }
}