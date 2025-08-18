#!/bin/bash
# AUJ Platform Backup and Recovery Script
# Early Decision System Removal - Risk Mitigation Implementation

# Configuration
BACKUP_DIR="./backups"
DATE_STAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_NAME="auj_platform"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Create backup directory
create_backup_dir() {
    log "Creating backup directory: $BACKUP_DIR/$DATE_STAMP"
    mkdir -p "$BACKUP_DIR/$DATE_STAMP"
    if [ $? -eq 0 ]; then
        log "Backup directory created successfully"
        return 0
    else
        error "Failed to create backup directory"
        return 1
    fi
}

# Git repository backup
backup_repository() {
    log "Creating git repository backup..."
    
    # Create backup branch
    git checkout -b "backup/before-early-decision-removal-$DATE_STAMP"
    if [ $? -eq 0 ]; then
        log "Backup branch created: backup/before-early-decision-removal-$DATE_STAMP"
    else
        error "Failed to create backup branch"
        return 1
    fi
    
    # Create archive of current state
    git archive --format=tar.gz --output="$BACKUP_DIR/$DATE_STAMP/auj_platform_source_$DATE_STAMP.tar.gz" HEAD
    if [ $? -eq 0 ]; then
        log "Source code archive created successfully"
    else
        error "Failed to create source code archive"
        return 1
    fi
    
    # Return to master branch
    git checkout master
    
    return 0
}

# Database backup
backup_databases() {
    log "Creating database backups..."
    
    # SQLite databases backup
    for db_file in $(find . -name "*.db" -type f); do
        db_name=$(basename "$db_file" .db)
        backup_file="$BACKUP_DIR/$DATE_STAMP/${db_name}_backup_$DATE_STAMP.db"
        
        log "Backing up database: $db_file -> $backup_file"
        cp "$db_file" "$backup_file"
        
        if [ $? -eq 0 ]; then
            log "Database backup successful: $db_name"
        else
            error "Database backup failed: $db_name"
            return 1
        fi
    done
    
    return 0
}

# Configuration backup
backup_configurations() {
    log "Creating configuration backups..."
    
    # Backup config directory
    config_backup="$BACKUP_DIR/$DATE_STAMP/config_backup_$DATE_STAMP"
    cp -r config/ "$config_backup"
    
    if [ $? -eq 0 ]; then
        log "Configuration backup successful"
    else
        error "Configuration backup failed"
        return 1
    fi
    
    # Backup additional configuration files
    for config_file in pyproject.toml setup_environment.* docker-compose*.yml; do
        if [ -f "$config_file" ]; then
            cp "$config_file" "$BACKUP_DIR/$DATE_STAMP/"
            log "Backed up: $config_file"
        fi
    done
    
    return 0
}

# Performance baseline backup
backup_performance_baseline() {
    log "Creating performance baseline backup..."
    
    # Create performance report
    performance_report="$BACKUP_DIR/$DATE_STAMP/performance_baseline_$DATE_STAMP.txt"
    
    cat > "$performance_report" << EOF
AUJ Platform Performance Baseline Report
========================================
Date: $(date)
Commit: $(git rev-parse HEAD)
Branch: $(git branch --show-current)

System Information:
- OS: $(uname -s)
- Architecture: $(uname -m)
- Python Version: $(python --version 2>&1)

Database Files:
$(find . -name "*.db" -type f -exec ls -lh {} \;)

Configuration Status:
$(grep -r "enable_early_decisions\|early_decision" config/ || echo "No early decision references found")

Git Status:
$(git status --porcelain)

Last 5 Commits:
$(git log --oneline -5)
EOF

    log "Performance baseline report created: $performance_report"
    return 0
}

# Validation of backup integrity
validate_backups() {
    log "Validating backup integrity..."
    
    backup_path="$BACKUP_DIR/$DATE_STAMP"
    
    # Check if all expected files exist
    required_files=(
        "auj_platform_source_$DATE_STAMP.tar.gz"
        "config_backup_$DATE_STAMP"
        "performance_baseline_$DATE_STAMP.txt"
    )
    
    for file in "${required_files[@]}"; do
        if [ -e "$backup_path/$file" ]; then
            log "✓ Backup file exists: $file"
        else
            error "✗ Missing backup file: $file"
            return 1
        fi
    done
    
    # Validate archive integrity
    tar -tzf "$backup_path/auj_platform_source_$DATE_STAMP.tar.gz" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        log "✓ Source archive integrity validated"
    else
        error "✗ Source archive integrity check failed"
        return 1
    fi
    
    log "All backup validations passed"
    return 0
}

# Main backup function
perform_complete_backup() {
    log "Starting comprehensive backup procedure..."
    
    create_backup_dir || return 1
    backup_repository || return 1
    backup_databases || return 1
    backup_configurations || return 1
    backup_performance_baseline || return 1
    validate_backups || return 1
    
    log "Comprehensive backup completed successfully!"
    log "Backup location: $BACKUP_DIR/$DATE_STAMP"
    
    return 0
}

# Rollback function
perform_rollback() {
    local backup_date="$1"
    
    if [ -z "$backup_date" ]; then
        error "Backup date required for rollback. Usage: $0 rollback YYYYMMDD_HHMMSS"
        return 1
    fi
    
    backup_path="$BACKUP_DIR/$backup_date"
    
    if [ ! -d "$backup_path" ]; then
        error "Backup directory not found: $backup_path"
        return 1
    fi
    
    warn "PERFORMING SYSTEM ROLLBACK - This will overwrite current state!"
    read -p "Are you sure? (yes/no): " confirmation
    
    if [ "$confirmation" != "yes" ]; then
        log "Rollback cancelled by user"
        return 1
    fi
    
    log "Starting rollback procedure..."
    
    # Stop any running services (add specific stop commands here)
    warn "Stop AUJ Platform services before continuing"
    
    # Restore configuration
    log "Restoring configuration..."
    rm -rf config/
    cp -r "$backup_path/config_backup_$backup_date" config/
    
    # Restore databases
    log "Restoring databases..."
    for backup_db in "$backup_path"/*.db; do
        if [ -f "$backup_db" ]; then
            db_name=$(basename "$backup_db" | sed "s/_backup_$backup_date\.db/.db/")
            cp "$backup_db" "$db_name"
            log "Restored database: $db_name"
        fi
    done
    
    # Restore source code (if needed)
    log "Restoring source code from git backup branch..."
    git checkout "backup/before-early-decision-removal-$backup_date"
    
    log "Rollback completed successfully!"
    warn "Please restart AUJ Platform services and validate system operation"
    
    return 0
}

# Display backup status
show_backup_status() {
    log "Available backups:"
    
    if [ -d "$BACKUP_DIR" ]; then
        for backup in "$BACKUP_DIR"/*; do
            if [ -d "$backup" ]; then
                backup_name=$(basename "$backup")
                backup_size=$(du -sh "$backup" | cut -f1)
                log "  $backup_name ($backup_size)"
            fi
        done
    else
        warn "No backup directory found"
    fi
}

# Main script logic
case "$1" in
    "backup")
        perform_complete_backup
        ;;
    "rollback")
        perform_rollback "$2"
        ;;
    "status")
        show_backup_status
        ;;
    *)
        echo "Usage: $0 {backup|rollback|status}"
        echo "  backup          - Perform comprehensive backup"
        echo "  rollback DATE   - Rollback to specific backup (YYYYMMDD_HHMMSS)"
        echo "  status          - Show available backups"
        exit 1
        ;;
esac

exit $?