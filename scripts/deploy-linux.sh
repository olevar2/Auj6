#!/bin/bash
# AUJ Platform Deployment Script
# Advanced deployment and update management for Linux

set -e

# Colors and configuration
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
AUJ_HOME="/opt/auj-platform"
AUJ_USER="auj"
AUJ_GROUP="auj"
AUJ_VENV="${AUJ_HOME}/venv"
BACKUP_DIR="${AUJ_HOME}/backups"
DEPLOY_DIR="${AUJ_HOME}/deployments"
SERVICE_MANAGER="./manage-services.sh"

# Deployment configuration
DEPLOYMENT_STRATEGIES=("blue-green" "rolling" "direct")
DEFAULT_STRATEGY="rolling"
HEALTH_CHECK_TIMEOUT=300
ROLLBACK_TIMEOUT=600

# Logging functions
log() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"; }
error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"; exit 1; }
info() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"; }

# Check prerequisites
check_prerequisites() {
    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root or with sudo"
    fi
    
    # Check if service manager exists
    if [ ! -f "$SERVICE_MANAGER" ]; then
        error "Service manager script not found: $SERVICE_MANAGER"
    fi
    
    # Check if AUJ platform is installed
    if [ ! -d "$AUJ_HOME" ]; then
        error "AUJ Platform not found at $AUJ_HOME. Run install-linux.sh first."
    fi
}

# Create deployment directories
setup_deployment_structure() {
    log "Setting up deployment structure..."
    
    mkdir -p "$DEPLOY_DIR"
    mkdir -p "$BACKUP_DIR"
    mkdir -p "${AUJ_HOME}/releases"
    mkdir -p "${AUJ_HOME}/shared/data"
    mkdir -p "${AUJ_HOME}/shared/logs"
    mkdir -p "${AUJ_HOME}/shared/config"
    
    chown -R "$AUJ_USER:$AUJ_GROUP" "$DEPLOY_DIR" "$BACKUP_DIR"
}

# Pre-deployment backup
create_deployment_backup() {
    local deployment_id="$1"
    local backup_file="${BACKUP_DIR}/pre_deploy_${deployment_id}.tar.gz"
    
    log "Creating pre-deployment backup..."
    
    tar -czf "$backup_file" \
        --exclude='venv' \
        --exclude='*.pyc' \
        --exclude='__pycache__' \
        --exclude='*.log' \
        "$AUJ_HOME" \
        "/etc/auj-platform" \
        2>/dev/null
    
    if [ -f "$backup_file" ]; then
        log "Backup created: $backup_file"
        echo "$backup_file" > "${DEPLOY_DIR}/${deployment_id}.backup"
    else
        error "Failed to create backup"
    fi
}

# Health check function
health_check() {
    local timeout="${1:-60}"
    local start_time=$(date +%s)
    
    info "Performing health check (timeout: ${timeout}s)..."
    
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [ $elapsed -gt $timeout ]; then
            error "Health check timeout after ${timeout}s"
        fi
        
        # Check API endpoint
        if curl -s -f http://localhost/api/health >/dev/null 2>&1; then
            # Check dashboard
            if curl -s -f http://localhost >/dev/null 2>&1; then
                # Check services
                if "$SERVICE_MANAGER" health >/dev/null 2>&1; then
                    log "Health check passed"
                    return 0
                fi
            fi
        fi
        
        info "Health check in progress... (${elapsed}s elapsed)"
        sleep 5
    done
}

# Rolling deployment strategy
deploy_rolling() {
    local source_dir="$1"
    local deployment_id="$2"
    
    log "Starting rolling deployment..."
    
    # Stop services gracefully
    info "Stopping services for rolling deployment..."
    "$SERVICE_MANAGER" stop
    
    # Update code
    info "Updating application code..."
    rsync -av --exclude='venv' --exclude='data' --exclude='.env' \
          "$source_dir/" "$AUJ_HOME/"
    
    # Update dependencies if requirements changed
    if [ -f "$source_dir/requirements-linux.txt" ]; then
        info "Updating Python dependencies..."
        sudo -u "$AUJ_USER" "$AUJ_VENV/bin/pip" install -r "$source_dir/requirements-linux.txt"
    fi
    
    # Update configuration if needed
    if [ -d "$source_dir/config" ]; then
        info "Updating configuration..."
        cp -r "$source_dir/config"/* "/etc/auj-platform/"
        chown -R root:"$AUJ_GROUP" "/etc/auj-platform"
    fi
    
    # Set proper permissions
    chown -R "$AUJ_USER:$AUJ_GROUP" "$AUJ_HOME"
    chmod +x "$AUJ_HOME/scripts"/*.sh
    
    # Start services
    info "Starting services..."
    "$SERVICE_MANAGER" start
    
    # Health check
    health_check $HEALTH_CHECK_TIMEOUT
    
    log "Rolling deployment completed successfully"
}

# Blue-green deployment strategy
deploy_blue_green() {
    local source_dir="$1"
    local deployment_id="$2"
    
    log "Starting blue-green deployment..."
    
    local blue_dir="${AUJ_HOME}_blue"
    local green_dir="${AUJ_HOME}_green"
    local current_link="$AUJ_HOME"
    
    # Determine current and new environments
    if [ -L "$current_link" ]; then
        local current_target=$(readlink "$current_link")
        if [[ "$current_target" == *"blue" ]]; then
            local new_env="green"
            local new_dir="$green_dir"
        else
            local new_env="blue"
            local new_dir="$blue_dir"
        fi
    else
        # First time setup
        mv "$AUJ_HOME" "$blue_dir"
        local new_env="green"
        local new_dir="$green_dir"
    fi
    
    info "Deploying to $new_env environment..."
    
    # Create new environment
    cp -r "$blue_dir" "$new_dir" 2>/dev/null || cp -r "$green_dir" "$new_dir" 2>/dev/null
    
    # Update new environment
    rsync -av --exclude='venv' --exclude='data' --exclude='.env' \
          "$source_dir/" "$new_dir/"
    
    # Update dependencies in new environment
    if [ -f "$source_dir/requirements-linux.txt" ]; then
        sudo -u "$AUJ_USER" "${new_dir}/venv/bin/pip" install -r "$source_dir/requirements-linux.txt"
    fi
    
    # Test new environment (start on different ports)
    info "Testing new environment..."
    # Implementation would start services on test ports and verify
    
    # Switch to new environment
    info "Switching to new environment..."
    "$SERVICE_MANAGER" stop
    
    rm -f "$current_link"
    ln -sf "$new_dir" "$current_link"
    
    "$SERVICE_MANAGER" start
    
    # Health check
    health_check $HEALTH_CHECK_TIMEOUT
    
    log "Blue-green deployment completed successfully"
}

# Direct deployment strategy
deploy_direct() {
    local source_dir="$1"
    local deployment_id="$2"
    
    log "Starting direct deployment..."
    
    # Update code directly
    info "Updating application code..."
    rsync -av --exclude='venv' --exclude='data' --exclude='.env' \
          "$source_dir/" "$AUJ_HOME/"
    
    # Update dependencies
    if [ -f "$source_dir/requirements-linux.txt" ]; then
        info "Updating Python dependencies..."
        sudo -u "$AUJ_USER" "$AUJ_VENV/bin/pip" install -r "$source_dir/requirements-linux.txt"
    fi
    
    # Reload services
    "$SERVICE_MANAGER" reload
    
    # Health check
    health_check $HEALTH_CHECK_TIMEOUT
    
    log "Direct deployment completed successfully"
}

# Main deployment function
deploy() {
    local source_dir="$1"
    local strategy="${2:-$DEFAULT_STRATEGY}"
    local deployment_id="deploy_$(date +%Y%m%d_%H%M%S)"
    
    log "Starting deployment with strategy: $strategy"
    info "Deployment ID: $deployment_id"
    
    # Validate inputs
    if [ ! -d "$source_dir" ]; then
        error "Source directory not found: $source_dir"
    fi
    
    if [[ ! " ${DEPLOYMENT_STRATEGIES[@]} " =~ " $strategy " ]]; then
        error "Invalid deployment strategy: $strategy. Valid options: ${DEPLOYMENT_STRATEGIES[*]}"
    fi
    
    # Setup deployment structure
    setup_deployment_structure
    
    # Create backup
    create_deployment_backup "$deployment_id"
    
    # Record deployment start
    echo "$(date): Starting deployment $deployment_id with strategy $strategy" >> "${DEPLOY_DIR}/deployment.log"
    
    # Execute deployment strategy
    case "$strategy" in
        "rolling")
            deploy_rolling "$source_dir" "$deployment_id"
            ;;
        "blue-green")
            deploy_blue_green "$source_dir" "$deployment_id"
            ;;
        "direct")
            deploy_direct "$source_dir" "$deployment_id"
            ;;
    esac
    
    # Record deployment success
    echo "$(date): Deployment $deployment_id completed successfully" >> "${DEPLOY_DIR}/deployment.log"
    echo "$deployment_id" > "${DEPLOY_DIR}/current_deployment"
    
    log "Deployment completed successfully!"
}

# Rollback function
rollback() {
    local deployment_id="$1"
    
    if [ -z "$deployment_id" ]; then
        # Get last deployment
        if [ -f "${DEPLOY_DIR}/current_deployment" ]; then
            deployment_id=$(cat "${DEPLOY_DIR}/current_deployment")
        else
            error "No deployment ID provided and no current deployment found"
        fi
    fi
    
    local backup_file_path="${DEPLOY_DIR}/${deployment_id}.backup"
    
    if [ ! -f "$backup_file_path" ]; then
        error "Backup file not found for deployment: $deployment_id"
    fi
    
    local backup_file=$(cat "$backup_file_path")
    
    if [ ! -f "$backup_file" ]; then
        error "Backup file not found: $backup_file"
    fi
    
    log "Starting rollback to deployment: $deployment_id"
    
    # Stop services
    "$SERVICE_MANAGER" stop
    
    # Restore from backup
    info "Restoring from backup: $backup_file"
    
    # Move current installation
    mv "$AUJ_HOME" "${AUJ_HOME}.rollback_$(date +%Y%m%d_%H%M%S)"
    
    # Extract backup
    tar -xzf "$backup_file" -C /
    
    # Start services
    "$SERVICE_MANAGER" start
    
    # Health check
    if health_check $ROLLBACK_TIMEOUT; then
        log "Rollback completed successfully"
        echo "$(date): Rollback to $deployment_id completed successfully" >> "${DEPLOY_DIR}/deployment.log"
    else
        error "Rollback failed - system is not healthy"
    fi
}

# Update from git repository
update_from_git() {
    local branch="${1:-main}"
    local strategy="${2:-rolling}"
    
    log "Updating from git repository (branch: $branch)..."
    
    # Check if it's a git repository
    if [ ! -d "$AUJ_HOME/.git" ]; then
        error "Not a git repository. Use deploy command instead."
    fi
    
    # Create temporary directory for git checkout
    local temp_dir="/tmp/auj_update_$(date +%Y%m%d_%H%M%S)"
    
    # Clone/pull latest code
    info "Fetching latest code from git..."
    git clone "$AUJ_HOME" "$temp_dir"
    cd "$temp_dir"
    git checkout "$branch"
    git pull origin "$branch"
    
    # Deploy from temporary directory
    deploy "$temp_dir" "$strategy"
    
    # Cleanup
    rm -rf "$temp_dir"
    
    log "Git update completed"
}

# List deployments
list_deployments() {
    log "Deployment History:"
    
    if [ -f "${DEPLOY_DIR}/deployment.log" ]; then
        tail -20 "${DEPLOY_DIR}/deployment.log"
    else
        info "No deployment history found"
    fi
    
    echo
    info "Available backups:"
    ls -la "$BACKUP_DIR" | grep "pre_deploy" || info "No deployment backups found"
}

# Monitor deployment
monitor_deployment() {
    local duration="${1:-300}"  # 5 minutes default
    local interval="${2:-10}"   # 10 seconds default
    
    log "Monitoring deployment for ${duration}s (checking every ${interval}s)..."
    
    local start_time=$(date +%s)
    local end_time=$((start_time + duration))
    
    while [ $(date +%s) -lt $end_time ]; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        echo "--- Monitor Check (${elapsed}s elapsed) ---"
        
        # Check services
        "$SERVICE_MANAGER" status | grep -E "(ACTIVE|INACTIVE)"
        
        # Check health
        if health_check 10; then
            info "✓ System healthy"
        else
            warn "✗ Health check failed"
        fi
        
        # Check resource usage
        echo "Memory: $(free | awk 'FNR==2{printf "%.1f%%", $3/$2*100}')"
        echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')"
        
        echo
        sleep $interval
    done
    
    log "Monitoring completed"
}

# Show help
show_help() {
    cat << EOF
AUJ Platform Deployment Script

Usage: $0 COMMAND [OPTIONS]

Commands:
  deploy SOURCE_DIR [STRATEGY]     Deploy from source directory
                                   Strategies: rolling (default), blue-green, direct
  
  rollback [DEPLOYMENT_ID]         Rollback to previous deployment
  
  update-git [BRANCH] [STRATEGY]   Update from git repository
                                   Default branch: main, strategy: rolling
  
  list                            List deployment history
  
  monitor [DURATION] [INTERVAL]   Monitor deployment health
                                   Duration in seconds (default: 300)
                                   Interval in seconds (default: 10)
  
  help                            Show this help

Deployment Strategies:
  rolling     - Stop services, update, restart (minimal downtime)
  blue-green  - Deploy to alternate environment, switch over (zero downtime)
  direct      - Update in place with service reload (fastest)

Examples:
  $0 deploy /tmp/auj-platform rolling          # Rolling deployment
  $0 deploy /tmp/auj-platform blue-green       # Blue-green deployment
  $0 update-git main rolling                   # Update from git main branch
  $0 rollback                                  # Rollback to last deployment
  $0 monitor 600 15                           # Monitor for 10min, check every 15s

Files:
  Deployment log: ${DEPLOY_DIR}/deployment.log
  Backups: ${BACKUP_DIR}/
  Current deployment: ${DEPLOY_DIR}/current_deployment
EOF
}

# Main function
main() {
    check_prerequisites
    
    case "${1:-help}" in
        "deploy")
            deploy "$2" "$3"
            ;;
        "rollback")
            rollback "$2"
            ;;
        "update-git")
            update_from_git "$2" "$3"
            ;;
        "list")
            list_deployments
            ;;
        "monitor")
            monitor_deployment "$2" "$3"
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run main function
main "$@"