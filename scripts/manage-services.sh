#!/bin/bash
# AUJ Platform Service Management Script
# Comprehensive service management for Linux deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
AUJ_SERVICES=("auj-platform" "auj-metaapi" "auj-dashboard" "redis" "nginx")
AUJ_LOG_DIR="/var/log/auj-platform"
AUJ_HOME="/opt/auj-platform"

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check if running as root or with sudo
check_permissions() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root or with sudo"
    fi
}

# Display service status
show_status() {
    echo "================================================================"
    echo "               AUJ Platform Service Status"
    echo "================================================================"
    echo
    
    for service in "${AUJ_SERVICES[@]}"; do
        echo "--- $service ---"
        if systemctl is-active --quiet "$service"; then
            echo -e "Status: ${GREEN}ACTIVE${NC}"
        else
            echo -e "Status: ${RED}INACTIVE${NC}"
        fi
        
        if systemctl is-enabled --quiet "$service"; then
            echo -e "Enabled: ${GREEN}YES${NC}"
        else
            echo -e "Enabled: ${YELLOW}NO${NC}"
        fi
        
        # Show memory usage if active
        if systemctl is-active --quiet "$service"; then
            MEMORY=$(systemctl show "$service" --property=MemoryCurrent --value 2>/dev/null)
            if [ "$MEMORY" != "0" ] && [ "$MEMORY" != "" ]; then
                MEMORY_MB=$((MEMORY / 1024 / 1024))
                echo "Memory: ${MEMORY_MB}MB"
            fi
        fi
        
        echo
    done
    
    # System resources
    echo "--- System Resources ---"
    echo "Memory Usage:"
    free -h | head -2
    echo
    echo "CPU Load:"
    uptime
    echo
    echo "Disk Usage:"
    df -h | grep -E "(Filesystem|/dev/)"
    echo
}

# Start all services
start_services() {
    log "Starting AUJ Platform services..."
    
    # Start in dependency order
    local services_order=("redis" "auj-platform" "auj-metaapi" "auj-dashboard" "nginx")
    
    for service in "${services_order[@]}"; do
        info "Starting $service..."
        
        if systemctl start "$service"; then
            log "$service started successfully"
        else
            error "Failed to start $service"
        fi
        
        # Wait a moment between services
        sleep 2
    done
    
    log "All services started successfully"
    
    # Show final status
    show_status
}

# Stop all services
stop_services() {
    log "Stopping AUJ Platform services..."
    
    # Stop in reverse dependency order
    local services_order=("nginx" "auj-dashboard" "auj-metaapi" "auj-platform" "redis")
    
    for service in "${services_order[@]}"; do
        info "Stopping $service..."
        
        if systemctl stop "$service"; then
            log "$service stopped successfully"
        else
            warn "Failed to stop $service (may not be running)"
        fi
    done
    
    log "All services stopped"
}

# Restart all services
restart_services() {
    log "Restarting AUJ Platform services..."
    
    stop_services
    sleep 5
    start_services
    
    log "All services restarted"
}

# Reload services (graceful restart)
reload_services() {
    log "Reloading AUJ Platform services..."
    
    local reload_services=("auj-platform" "auj-metaapi" "nginx")
    
    for service in "${reload_services[@]}"; do
        info "Reloading $service..."
        
        if systemctl reload "$service" 2>/dev/null; then
            log "$service reloaded successfully"
        else
            info "$service doesn't support reload, restarting instead..."
            systemctl restart "$service"
            log "$service restarted successfully"
        fi
    done
    
    log "Services reloaded"
}

# Enable all services for auto-start
enable_services() {
    log "Enabling AUJ Platform services for auto-start..."
    
    for service in "${AUJ_SERVICES[@]}"; do
        info "Enabling $service..."
        
        if systemctl enable "$service"; then
            log "$service enabled successfully"
        else
            error "Failed to enable $service"
        fi
    done
    
    log "All services enabled for auto-start"
}

# Disable services
disable_services() {
    log "Disabling AUJ Platform services..."
    
    for service in "${AUJ_SERVICES[@]}"; do
        info "Disabling $service..."
        
        if systemctl disable "$service"; then
            log "$service disabled successfully"
        else
            warn "Failed to disable $service"
        fi
    done
    
    log "All services disabled"
}

# Show logs for a specific service
show_logs() {
    local service=${1:-"auj-platform"}
    local lines=${2:-50}
    
    info "Showing last $lines lines of $service logs..."
    
    if systemctl list-units --full --all | grep -q "$service.service"; then
        journalctl -u "$service" -n "$lines" --no-pager
    else
        error "Service $service not found"
    fi
}

# Follow logs for a specific service
follow_logs() {
    local service=${1:-"auj-platform"}
    
    info "Following logs for $service (Ctrl+C to stop)..."
    
    if systemctl list-units --full --all | grep -q "$service.service"; then
        journalctl -u "$service" -f
    else
        error "Service $service not found"
    fi
}

# Show all recent logs
show_all_logs() {
    local lines=${1:-20}
    
    info "Showing recent logs from all AUJ services..."
    
    for service in "${AUJ_SERVICES[@]}"; do
        echo "--- $service (last $lines lines) ---"
        journalctl -u "$service" -n "$lines" --no-pager 2>/dev/null || echo "No logs available"
        echo
    done
}

# Health check
health_check() {
    log "Performing health check..."
    
    local healthy=true
    
    # Check services
    for service in "${AUJ_SERVICES[@]}"; do
        if systemctl is-active --quiet "$service"; then
            info "✓ $service is running"
        else
            warn "✗ $service is not running"
            healthy=false
        fi
    done
    
    # Check API endpoint
    if curl -s -f http://localhost/api/health >/dev/null 2>&1; then
        info "✓ API endpoint is responding"
    else
        warn "✗ API endpoint is not responding"
        healthy=false
    fi
    
    # Check dashboard
    if curl -s -f http://localhost >/dev/null 2>&1; then
        info "✓ Dashboard is responding"
    else
        warn "✗ Dashboard is not responding"
        healthy=false
    fi
    
    # Check system resources
    local memory_usage=$(free | awk 'FNR==2{printf "%.2f", $3/$2*100}')
    local disk_usage=$(df / | awk 'FNR==2{printf "%.2f", $5}' | sed 's/%//')
    
    if (( $(echo "$memory_usage > 90" | bc -l) )); then
        warn "✗ High memory usage: ${memory_usage}%"
        healthy=false
    else
        info "✓ Memory usage: ${memory_usage}%"
    fi
    
    if (( $(echo "$disk_usage > 90" | bc -l) )); then
        warn "✗ High disk usage: ${disk_usage}%"
        healthy=false
    else
        info "✓ Disk usage: ${disk_usage}%"
    fi
    
    if $healthy; then
        log "Health check passed - all systems operational"
        return 0
    else
        error "Health check failed - issues detected"
        return 1
    fi
}

# Clean up logs
cleanup_logs() {
    local days=${1:-30}
    
    log "Cleaning up logs older than $days days..."
    
    # Clean journalctl logs
    journalctl --vacuum-time=${days}d
    
    # Clean application logs
    find "$AUJ_LOG_DIR" -name "*.log" -mtime +$days -delete 2>/dev/null || true
    find "$AUJ_LOG_DIR" -name "*.log.*" -mtime +$days -delete 2>/dev/null || true
    
    log "Log cleanup completed"
}

# Update services
update_services() {
    log "Updating AUJ Platform..."
    
    # Stop services
    stop_services
    
    # Update code (if git repository)
    if [ -d "$AUJ_HOME/.git" ]; then
        info "Updating code from git repository..."
        cd "$AUJ_HOME"
        sudo -u auj git pull
        
        # Update Python dependencies
        info "Updating Python dependencies..."
        sudo -u auj "$AUJ_HOME/venv/bin/pip" install --upgrade -r requirements-linux.txt
    else
        warn "Not a git repository - manual update required"
    fi
    
    # Reload systemd
    systemctl daemon-reload
    
    # Start services
    start_services
    
    log "Update completed"
}

# Backup system
backup_system() {
    local backup_dir="$AUJ_HOME/backups"
    local date=$(date +%Y%m%d_%H%M%S)
    local backup_file="auj_backup_$date.tar.gz"
    
    log "Creating system backup..."
    
    mkdir -p "$backup_dir"
    
    # Create backup
    tar -czf "$backup_dir/$backup_file" \
        --exclude='*.pyc' \
        --exclude='__pycache__' \
        --exclude='venv' \
        --exclude='*.log' \
        "$AUJ_HOME/data" \
        "/etc/auj-platform" \
        "$AUJ_HOME/.env" \
        2>/dev/null || true
    
    if [ -f "$backup_dir/$backup_file" ]; then
        log "Backup created: $backup_dir/$backup_file"
        
        # Keep only last 7 backups
        find "$backup_dir" -name "auj_backup_*.tar.gz" -mtime +7 -delete 2>/dev/null || true
        
        log "Old backups cleaned up"
    else
        error "Failed to create backup"
    fi
}

# Show help
show_help() {
    cat << EOF
AUJ Platform Service Management Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
  status              Show status of all services
  start               Start all services
  stop                Stop all services
  restart             Restart all services
  reload              Reload services (graceful restart)
  enable              Enable services for auto-start
  disable             Disable services
  
  logs [SERVICE] [LINES]    Show logs for service (default: auj-platform, 50 lines)
  follow [SERVICE]          Follow logs for service (default: auj-platform)
  alllogs [LINES]          Show recent logs from all services (default: 20 lines)
  
  health              Perform health check
  cleanup [DAYS]      Clean up logs older than X days (default: 30)
  update              Update platform and restart services
  backup              Create system backup
  
  help                Show this help message

Examples:
  $0 status                    # Show service status
  $0 logs auj-platform 100    # Show last 100 lines of main service
  $0 follow auj-metaapi       # Follow MetaApi service logs
  $0 cleanup 7                # Clean logs older than 7 days
  $0 health                   # Check system health

Services managed:
  - auj-platform      Main trading platform service
  - auj-metaapi       MetaApi integration service  
  - auj-dashboard     Web dashboard service
  - redis             Redis cache service
  - nginx             Web server and reverse proxy

Logs location: $AUJ_LOG_DIR
Configuration: /etc/auj-platform
Data directory: $AUJ_HOME/data
EOF
}

# Main function
main() {
    check_permissions
    
    case "${1:-help}" in
        "status")
            show_status
            ;;
        "start")
            start_services
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            restart_services
            ;;
        "reload")
            reload_services
            ;;
        "enable")
            enable_services
            ;;
        "disable")
            disable_services
            ;;
        "logs")
            show_logs "$2" "$3"
            ;;
        "follow")
            follow_logs "$2"
            ;;
        "alllogs")
            show_all_logs "$2"
            ;;
        "health")
            health_check
            ;;
        "cleanup")
            cleanup_logs "$2"
            ;;
        "update")
            update_services
            ;;
        "backup")
            backup_system
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run main function with all arguments
main "$@"