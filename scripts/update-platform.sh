#!/bin/bash

# AUJ Platform Update Script for Linux
# Handles updates, patches, and maintenance tasks

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="/opt/auj-platform"
LOG_DIR="/var/log/auj-platform"
CONFIG_DIR="/etc/auj-platform"
BACKUP_DIR="/opt/auj-platform/backups"
VENV_DIR="$INSTALL_DIR/venv"
PLATFORM_USER="auj"

# Update settings
UPDATE_LOG="$LOG_DIR/update-$(date +%Y%m%d-%H%M%S).log"
BACKUP_BEFORE_UPDATE=true
AUTO_RESTART_SERVICES=true

# Utility functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$UPDATE_LOG"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$UPDATE_LOG"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$UPDATE_LOG"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}" | tee -a "$UPDATE_LOG"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root (use sudo)"
    fi
}

# Create backup before update
create_backup() {
    if [[ "$BACKUP_BEFORE_UPDATE" == "true" ]]; then
        local backup_timestamp=$(date +%Y%m%d-%H%M%S)
        local backup_path="$BACKUP_DIR/pre-update-$backup_timestamp"
        
        log "Creating backup before update..."
        mkdir -p "$backup_path"
        
        # Backup database
        if [[ -f "$INSTALL_DIR/data/auj_platform.db" ]]; then
            cp "$INSTALL_DIR/data/auj_platform.db" "$backup_path/"
            log "Database backed up"
        fi
        
        # Backup configuration
        cp -r "$CONFIG_DIR" "$backup_path/config" 2>/dev/null || true
        cp -r "$INSTALL_DIR/config" "$backup_path/app-config" 2>/dev/null || true
        
        # Backup virtual environment package list
        sudo -u "$PLATFORM_USER" "$VENV_DIR/bin/pip" freeze > "$backup_path/requirements-snapshot.txt"
        
        # Set ownership
        chown -R "$PLATFORM_USER:$PLATFORM_USER" "$backup_path"
        
        log "Backup created at: $backup_path"
        export BACKUP_PATH="$backup_path"
    fi
}

# Update system packages
update_system_packages() {
    log "Updating system packages..."
    
    # Detect distribution
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        DISTRO=$ID
    else
        error "Cannot detect Linux distribution"
    fi
    
    case $DISTRO in
        ubuntu|debian)
            apt-get update
            apt-get upgrade -y
            apt-get autoremove -y
            apt-get autoclean
            ;;
        centos|rhel|fedora)
            if command -v dnf &> /dev/null; then
                dnf update -y
                dnf autoremove -y
                dnf clean all
            else
                yum update -y
                yum autoremove -y
                yum clean all
            fi
            ;;
        *)
            warn "Unsupported distribution: $DISTRO"
            ;;
    esac
    
    log "System packages updated"
}

# Update Python dependencies
update_python_dependencies() {
    log "Updating Python dependencies..."
    
    # Stop services before updating
    if [[ "$AUTO_RESTART_SERVICES" == "true" ]]; then
        systemctl stop auj-platform || true
    fi
    
    # Update pip
    sudo -u "$PLATFORM_USER" "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel
    
    # Update requirements
    if [[ -f "$INSTALL_DIR/requirements.txt" ]]; then
        sudo -u "$PLATFORM_USER" "$VENV_DIR/bin/pip" install --upgrade -r "$INSTALL_DIR/requirements.txt"
    fi
    
    if [[ -f "$INSTALL_DIR/requirements-linux.txt" ]]; then
        sudo -u "$PLATFORM_USER" "$VENV_DIR/bin/pip" install --upgrade -r "$INSTALL_DIR/requirements-linux.txt"
    fi
    
    # Update the platform package itself
    if [[ -f "$INSTALL_DIR/setup.py" ]] || [[ -f "$INSTALL_DIR/pyproject.toml" ]]; then
        sudo -u "$PLATFORM_USER" "$VENV_DIR/bin/pip" install --upgrade -e "$INSTALL_DIR"
    fi
    
    log "Python dependencies updated"
}

# Update configuration files
update_configuration() {
    log "Checking configuration updates..."
    
    # Check if there are new configuration templates
    if [[ -f "$INSTALL_DIR/config/.env.template" ]] && [[ -f "$CONFIG_DIR/env.template" ]]; then
        if ! diff -q "$INSTALL_DIR/config/.env.template" "$CONFIG_DIR/env.template" >/dev/null 2>&1; then
            warn "New environment template available"
            cp "$INSTALL_DIR/config/.env.template" "$CONFIG_DIR/env.template.new"
            info "New template saved as $CONFIG_DIR/env.template.new"
            info "Please review and merge changes into your .env file"
        fi
    fi
    
    # Copy new configuration files if they don't exist
    for config_file in "$INSTALL_DIR/config"/*.yaml; do
        if [[ -f "$config_file" ]]; then
            filename=$(basename "$config_file")
            if [[ ! -f "$CONFIG_DIR/$filename" ]]; then
                cp "$config_file" "$CONFIG_DIR/"
                log "New configuration file added: $filename"
            fi
        fi
    done
    
    log "Configuration checked"
}

# Check for database migrations
check_database_migrations() {
    log "Checking for database migrations..."
    
    # Check if alembic is available and there are migrations
    if sudo -u "$PLATFORM_USER" "$VENV_DIR/bin/python" -c "import alembic" 2>/dev/null; then
        cd "$INSTALL_DIR"
        if [[ -d "alembic" ]] || [[ -f "alembic.ini" ]]; then
            log "Running database migrations..."
            sudo -u "$PLATFORM_USER" "$VENV_DIR/bin/alembic" upgrade head
            log "Database migrations completed"
        else
            info "No database migrations found"
        fi
    else
        info "Alembic not available, skipping database migrations"
    fi
}

# Update SSL certificates
update_ssl_certificates() {
    log "Checking SSL certificates..."
    
    # Check if certificates exist and are expiring soon
    if [[ -f "/etc/ssl/certs/auj-platform.crt" ]]; then
        CERT_EXPIRY=$(openssl x509 -in /etc/ssl/certs/auj-platform.crt -noout -enddate | cut -d= -f2)
        CERT_EXPIRY_EPOCH=$(date -d "$CERT_EXPIRY" +%s)
        CURRENT_EPOCH=$(date +%s)
        DAYS_UNTIL_EXPIRY=$(( (CERT_EXPIRY_EPOCH - CURRENT_EPOCH) / 86400 ))
        
        if [[ $DAYS_UNTIL_EXPIRY -lt 30 ]]; then
            warn "SSL certificate expires in $DAYS_UNTIL_EXPIRY days"
            info "Consider renewing SSL certificate"
        else
            log "SSL certificate valid for $DAYS_UNTIL_EXPIRY days"
        fi
    fi
    
    # Try to renew Let's Encrypt certificates if certbot is available
    if command -v certbot &> /dev/null; then
        log "Attempting to renew Let's Encrypt certificates..."
        certbot renew --quiet || warn "Certificate renewal failed or not needed"
    fi
}

# Clean up old files
cleanup_old_files() {
    log "Cleaning up old files..."
    
    # Clean old log files (older than 30 days)
    find "$LOG_DIR" -name "*.log" -mtime +30 -delete 2>/dev/null || true
    
    # Clean old backup files (older than 90 days)
    find "$BACKUP_DIR" -type d -name "pre-update-*" -mtime +90 -exec rm -rf {} + 2>/dev/null || true
    find "$BACKUP_DIR" -type d -name "manual-*" -mtime +90 -exec rm -rf {} + 2>/dev/null || true
    
    # Clean pip cache
    sudo -u "$PLATFORM_USER" "$VENV_DIR/bin/pip" cache purge 2>/dev/null || true
    
    # Clean APT cache (if applicable)
    if command -v apt-get &> /dev/null; then
        apt-get autoclean
    fi
    
    log "Cleanup completed"
}

# Restart services
restart_services() {
    if [[ "$AUTO_RESTART_SERVICES" == "true" ]]; then
        log "Restarting services..."
        
        # Reload systemd
        systemctl daemon-reload
        
        # Restart services in order
        systemctl restart auj-platform || warn "Failed to restart auj-platform"
        systemctl restart nginx || warn "Failed to restart nginx"
        systemctl restart supervisor || warn "Failed to restart supervisor"
        
        # Wait for services to start
        sleep 10
        
        # Check if services are running
        if systemctl is-active --quiet auj-platform; then
            log "AUJ Platform service restarted successfully"
        else
            error "AUJ Platform service failed to start after update"
        fi
        
        log "Services restarted"
    fi
}

# Verify update
verify_update() {
    log "Verifying update..."
    
    # Check if main application is accessible
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        log "API health check passed"
    else
        warn "API health check failed"
    fi
    
    # Check if dashboard is accessible
    if curl -sf http://localhost:8501 >/dev/null 2>&1; then
        log "Dashboard health check passed"
    else
        warn "Dashboard health check failed"
    fi
    
    # Check Python packages
    log "Checking key package versions:"
    sudo -u "$PLATFORM_USER" "$VENV_DIR/bin/pip" list | grep -E "(metaapi|fastapi|pandas|numpy)" | head -5
    
    log "Update verification completed"
}

# Rollback function
rollback() {
    if [[ -n "${BACKUP_PATH:-}" ]] && [[ -d "$BACKUP_PATH" ]]; then
        warn "Rolling back changes..."
        
        # Stop services
        systemctl stop auj-platform || true
        
        # Restore database
        if [[ -f "$BACKUP_PATH/auj_platform.db" ]]; then
            cp "$BACKUP_PATH/auj_platform.db" "$INSTALL_DIR/data/"
            log "Database restored"
        fi
        
        # Restore configuration
        if [[ -d "$BACKUP_PATH/config" ]]; then
            cp -r "$BACKUP_PATH/config"/* "$CONFIG_DIR/"
            log "Configuration restored"
        fi
        
        # Restore Python packages
        if [[ -f "$BACKUP_PATH/requirements-snapshot.txt" ]]; then
            sudo -u "$PLATFORM_USER" "$VENV_DIR/bin/pip" install -r "$BACKUP_PATH/requirements-snapshot.txt"
            log "Python packages restored"
        fi
        
        # Restart services
        systemctl start auj-platform
        
        log "Rollback completed"
    else
        error "No backup found for rollback"
    fi
}

# Security updates only
security_update() {
    log "Performing security-only update..."
    
    case $DISTRO in
        ubuntu|debian)
            apt-get update
            apt-get upgrade -y --only-upgrade \
                $(apt list --upgradable 2>/dev/null | grep -i security | cut -d/ -f1)
            ;;
        centos|rhel|fedora)
            if command -v dnf &> /dev/null; then
                dnf update --security -y
            else
                yum update --security -y
            fi
            ;;
        *)
            warn "Security-only updates not supported for $DISTRO"
            ;;
    esac
    
    log "Security updates completed"
}

# Display update information
show_update_info() {
    echo
    echo "=== AUJ Platform Update Information ==="
    echo
    echo "Current Version:"
    sudo -u "$PLATFORM_USER" "$VENV_DIR/bin/python" -c "
import pkg_resources
try:
    version = pkg_resources.get_distribution('auj-platform').version
    print(f'AUJ Platform: {version}')
except:
    print('AUJ Platform: Version not found')
"
    echo
    echo "Key Dependencies:"
    sudo -u "$PLATFORM_USER" "$VENV_DIR/bin/pip" list | grep -E "(metaapi|fastapi|pandas|numpy|streamlit)" | head -10
    echo
    echo "Last Update: $(stat -c %y "$UPDATE_LOG" 2>/dev/null || echo "Never")"
    echo "Update Log: $UPDATE_LOG"
    echo
}

# Usage information
usage() {
    echo "AUJ Platform Update Script"
    echo
    echo "Usage: $0 <command> [options]"
    echo
    echo "Commands:"
    echo "  full               Perform full update (system + Python + config)"
    echo "  system             Update system packages only"
    echo "  python             Update Python dependencies only"
    echo "  security           Security updates only"
    echo "  config             Update configuration files only"
    echo "  ssl                Update SSL certificates"
    echo "  cleanup            Clean up old files and caches"
    echo "  verify             Verify current installation"
    echo "  rollback           Rollback to previous backup"
    echo "  info               Show update information"
    echo
    echo "Options:"
    echo "  --no-backup        Skip creating backup before update"
    echo "  --no-restart       Don't restart services after update"
    echo
    echo "Examples:"
    echo "  $0 full"
    echo "  $0 python --no-restart"
    echo "  $0 security"
}

# Main function
main() {
    local command="${1:-}"
    
    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-backup)
                BACKUP_BEFORE_UPDATE=false
                shift
                ;;
            --no-restart)
                AUTO_RESTART_SERVICES=false
                shift
                ;;
            *)
                if [[ -z "$command" ]]; then
                    command="$1"
                fi
                shift
                ;;
        esac
    done
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    case "$command" in
        "full")
            check_root
            log "Starting full update..."
            create_backup
            update_system_packages
            update_python_dependencies
            update_configuration
            check_database_migrations
            update_ssl_certificates
            cleanup_old_files
            restart_services
            verify_update
            log "Full update completed successfully"
            ;;
        "system")
            check_root
            create_backup
            update_system_packages
            log "System update completed"
            ;;
        "python")
            check_root
            create_backup
            update_python_dependencies
            check_database_migrations
            restart_services
            verify_update
            log "Python update completed"
            ;;
        "security")
            check_root
            security_update
            log "Security update completed"
            ;;
        "config")
            check_root
            update_configuration
            restart_services
            log "Configuration update completed"
            ;;
        "ssl")
            check_root
            update_ssl_certificates
            log "SSL update completed"
            ;;
        "cleanup")
            check_root
            cleanup_old_files
            log "Cleanup completed"
            ;;
        "verify")
            verify_update
            ;;
        "rollback")
            check_root
            rollback
            ;;
        "info")
            show_update_info
            ;;
        "help"|"-h"|"--help")
            usage
            ;;
        *)
            error "Unknown command: $command"
            echo
            usage
            exit 1
            ;;
    esac
}

# Trap for cleanup on exit
trap 'if [[ $? -ne 0 ]] && [[ -n "${BACKUP_PATH:-}" ]]; then warn "Update failed. Use '$0 rollback' to restore previous state"; fi' EXIT

# Run main function
main "$@"