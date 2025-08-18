#!/bin/bash
# AUJ Platform Monitoring and Diagnostics Script
# Comprehensive monitoring and troubleshooting for Linux deployment

set -e

# Colors and configuration
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
AUJ_HOME="/opt/auj-platform"
AUJ_LOG_DIR="/var/log/auj-platform"
AUJ_CONFIG_DIR="/etc/auj-platform"
AUJ_SERVICES=("auj-platform" "auj-metaapi" "auj-dashboard" "redis" "nginx")
MONITORING_INTERVAL=60
ALERT_THRESHOLDS=(
    "memory:85"
    "cpu:80" 
    "disk:90"
    "load:4.0"
)

# Logging functions
log() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"; }
error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"; }
info() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"; }
debug() { echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] DEBUG: $1${NC}"; }

# System information
get_system_info() {
    echo "================================================================"
    echo "                  AUJ Platform System Information"
    echo "================================================================"
    echo
    
    echo "--- System Details ---"
    echo "Hostname: $(hostname)"
    echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
    echo "Kernel: $(uname -r)"
    echo "Architecture: $(uname -m)"
    echo "Uptime: $(uptime -p)"
    echo
    
    echo "--- Hardware Information ---"
    echo "CPU: $(lscpu | grep "Model name" | cut -d':' -f2 | xargs)"
    echo "CPU Cores: $(nproc)"
    echo "Memory: $(free -h | awk 'NR==2{print $2}')"
    echo "Disk Space: $(df -h / | awk 'NR==2{print $2}')"
    echo
    
    echo "--- Network Information ---"
    echo "IP Address: $(hostname -I | awk '{print $1}')"
    echo "DNS Servers: $(cat /etc/resolv.conf | grep nameserver | awk '{print $2}' | tr '\n' ' ')"
    echo
}

# Service status overview
get_service_status() {
    echo "--- Service Status Overview ---"
    printf "%-20s %-10s %-10s %-15s %-10s\n" "SERVICE" "STATUS" "ENABLED" "MEMORY(MB)" "CPU%"
    echo "--------------------------------------------------------------------------------"
    
    for service in "${AUJ_SERVICES[@]}"; do
        # Service status
        if systemctl is-active --quiet "$service"; then
            status="${GREEN}ACTIVE${NC}"
        else
            status="${RED}INACTIVE${NC}"
        fi
        
        # Enabled status
        if systemctl is-enabled --quiet "$service"; then
            enabled="${GREEN}YES${NC}"
        else
            enabled="${YELLOW}NO${NC}"
        fi
        
        # Memory usage
        memory=$(systemctl show "$service" --property=MemoryCurrent --value 2>/dev/null)
        if [ "$memory" != "0" ] && [ "$memory" != "" ]; then
            memory_mb=$((memory / 1024 / 1024))
        else
            memory_mb="N/A"
        fi
        
        # CPU usage (approximate)
        cpu_usage=$(ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%cpu | grep -v grep | grep "$service" | awk '{sum += $5} END {printf "%.1f", sum}')
        [ -z "$cpu_usage" ] && cpu_usage="0.0"
        
        printf "%-20s %-20s %-20s %-15s %-10s\n" "$service" "$status" "$enabled" "$memory_mb" "$cpu_usage"
    done
    echo
}

# Resource monitoring
get_resource_usage() {
    echo "--- Resource Usage ---"
    
    # Memory usage
    memory_info=$(free -m | awk 'NR==2{printf "Used: %dMB (%.1f%%), Free: %dMB", $3, $3*100/$2, $4}')
    echo "Memory: $memory_info"
    
    # CPU usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
    echo "CPU Usage: $cpu_usage"
    
    # Load average
    load_avg=$(uptime | awk -F'load average:' '{print $2}')
    echo "Load Average:$load_avg"
    
    # Disk usage
    echo "Disk Usage:"
    df -h | grep -E "(Filesystem|/dev/)" | awk '{printf "  %-20s %8s %8s %8s %6s %s\n", $1, $2, $3, $4, $5, $6}'
    
    echo
}

# Network connectivity checks
check_network() {
    echo "--- Network Connectivity ---"
    
    # Check internet connectivity
    if ping -c 1 8.8.8.8 >/dev/null 2>&1; then
        echo -e "Internet: ${GREEN}✓ Connected${NC}"
    else
        echo -e "Internet: ${RED}✗ No connectivity${NC}"
    fi
    
    # Check DNS resolution
    if nslookup google.com >/dev/null 2>&1; then
        echo -e "DNS: ${GREEN}✓ Working${NC}"
    else
        echo -e "DNS: ${RED}✗ Failed${NC}"
    fi
    
    # Check MetaApi connectivity
    if curl -s --connect-timeout 5 https://mt-client-api-v1.new-york.agiliumtrade.agiliumlabs.cloud/health >/dev/null 2>&1; then
        echo -e "MetaApi: ${GREEN}✓ Reachable${NC}"
    else
        echo -e "MetaApi: ${YELLOW}? Unreachable${NC}"
    fi
    
    # Check local API
    if curl -s -f http://localhost/api/health >/dev/null 2>&1; then
        echo -e "Local API: ${GREEN}✓ Responding${NC}"
    else
        echo -e "Local API: ${RED}✗ Not responding${NC}"
    fi
    
    # Check dashboard
    if curl -s -f http://localhost >/dev/null 2>&1; then
        echo -e "Dashboard: ${GREEN}✓ Responding${NC}"
    else
        echo -e "Dashboard: ${RED}✗ Not responding${NC}"
    fi
    
    echo
}

# Database status
check_database() {
    echo "--- Database Status ---"
    
    local db_file="$AUJ_HOME/data/auj_platform.db"
    
    if [ -f "$db_file" ]; then
        local db_size=$(du -h "$db_file" | cut -f1)
        echo -e "Database File: ${GREEN}✓ Exists${NC} (Size: $db_size)"
        
        # Test database connection
        if sqlite3 "$db_file" "SELECT 1;" >/dev/null 2>&1; then
            echo -e "Database Access: ${GREEN}✓ OK${NC}"
            
            # Get table count
            local table_count=$(sqlite3 "$db_file" ".tables" | wc -w)
            echo "Tables: $table_count"
            
        else
            echo -e "Database Access: ${RED}✗ Failed${NC}"
        fi
    else
        echo -e "Database File: ${RED}✗ Not found${NC}"
    fi
    
    echo
}

# Configuration validation
check_configuration() {
    echo "--- Configuration Status ---"
    
    # Check main config
    if [ -f "$AUJ_CONFIG_DIR/main_config.yaml" ]; then
        echo -e "Main Config: ${GREEN}✓ Found${NC}"
        
        # Validate YAML syntax
        if python3 -c "import yaml; yaml.safe_load(open('$AUJ_CONFIG_DIR/main_config.yaml'))" 2>/dev/null; then
            echo -e "Config Syntax: ${GREEN}✓ Valid${NC}"
        else
            echo -e "Config Syntax: ${RED}✗ Invalid YAML${NC}"
        fi
    else
        echo -e "Main Config: ${RED}✗ Not found${NC}"
    fi
    
    # Check environment file
    if [ -f "$AUJ_HOME/.env" ]; then
        echo -e "Environment File: ${GREEN}✓ Found${NC}"
        
        # Check critical variables
        source "$AUJ_HOME/.env" 2>/dev/null
        
        if [ -n "$AUJ_METAAPI_TOKEN" ]; then
            echo -e "MetaApi Token: ${GREEN}✓ Set${NC}"
        else
            echo -e "MetaApi Token: ${RED}✗ Missing${NC}"
        fi
        
        if [ -n "$AUJ_METAAPI_ACCOUNT_ID" ]; then
            echo -e "MetaApi Account: ${GREEN}✓ Set${NC}"
        else
            echo -e "MetaApi Account: ${RED}✗ Missing${NC}"
        fi
    else
        echo -e "Environment File: ${RED}✗ Not found${NC}"
    fi
    
    echo
}

# Log analysis
analyze_logs() {
    local hours="${1:-1}"
    echo "--- Log Analysis (Last $hours hour(s)) ---"
    
    local since_time="${hours} hours ago"
    
    # Error count by service
    for service in "${AUJ_SERVICES[@]}"; do
        local error_count=$(journalctl -u "$service" --since "$since_time" | grep -i error | wc -l)
        local warn_count=$(journalctl -u "$service" --since "$since_time" | grep -i warning | wc -l)
        
        if [ "$error_count" -gt 0 ] || [ "$warn_count" -gt 0 ]; then
            printf "%-20s Errors: %-5s Warnings: %-5s\n" "$service" "$error_count" "$warn_count"
        fi
    done
    
    # Recent critical errors
    echo
    echo "Recent Critical Errors:"
    journalctl --since "$since_time" --priority=err --no-pager | tail -5 || echo "No critical errors found"
    
    echo
}

# Performance metrics
get_performance_metrics() {
    echo "--- Performance Metrics ---"
    
    # System load trend
    echo "Load Average Trend (1m, 5m, 15m):"
    uptime | awk -F'load average:' '{print "  " $2}'
    
    # Memory usage breakdown
    echo
    echo "Memory Usage Breakdown:"
    free -h | awk 'NR==2{printf "  Total: %s, Used: %s (%.1f%%), Free: %s, Available: %s\n", $2, $3, $3*100/$2, $4, $7}'
    
    # Top processes by CPU
    echo
    echo "Top CPU Consumers:"
    ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%cpu | head -6
    
    # Top processes by Memory
    echo
    echo "Top Memory Consumers:"
    ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%mem | head -6
    
    # I/O statistics
    if command -v iostat >/dev/null 2>&1; then
        echo
        echo "Disk I/O Statistics:"
        iostat -h 1 1 | tail -n +4
    fi
    
    echo
}

# Security check
security_check() {
    echo "--- Security Status ---"
    
    # Check firewall status
    if command -v ufw >/dev/null 2>&1; then
        if ufw status | grep -q "Status: active"; then
            echo -e "Firewall (UFW): ${GREEN}✓ Active${NC}"
        else
            echo -e "Firewall (UFW): ${YELLOW}! Inactive${NC}"
        fi
    elif command -v firewall-cmd >/dev/null 2>&1; then
        if systemctl is-active --quiet firewalld; then
            echo -e "Firewall (firewalld): ${GREEN}✓ Active${NC}"
        else
            echo -e "Firewall (firewalld): ${YELLOW}! Inactive${NC}"
        fi
    else
        echo -e "Firewall: ${RED}✗ Not found${NC}"
    fi
    
    # Check for unauthorized access attempts
    local ssh_fails=$(journalctl --since "24 hours ago" | grep -i "failed password" | wc -l)
    if [ "$ssh_fails" -gt 0 ]; then
        echo -e "SSH Failed Logins (24h): ${YELLOW}$ssh_fails attempts${NC}"
    else
        echo -e "SSH Failed Logins (24h): ${GREEN}✓ None${NC}"
    fi
    
    # Check file permissions
    local config_perms=$(stat -c "%a" "$AUJ_CONFIG_DIR" 2>/dev/null)
    if [ "$config_perms" = "750" ]; then
        echo -e "Config Permissions: ${GREEN}✓ Secure${NC}"
    else
        echo -e "Config Permissions: ${YELLOW}! Check required${NC} (current: $config_perms)"
    fi
    
    echo
}

# Comprehensive health check
comprehensive_health_check() {
    echo "================================================================"
    echo "              AUJ Platform Comprehensive Health Check"
    echo "================================================================"
    echo
    
    get_system_info
    get_service_status
    get_resource_usage
    check_network
    check_database
    check_configuration
    analyze_logs 1
    get_performance_metrics
    security_check
    
    # Health score calculation
    local health_score=100
    local issues=()
    
    # Check for critical issues
    for service in "${AUJ_SERVICES[@]}"; do
        if ! systemctl is-active --quiet "$service"; then
            health_score=$((health_score - 15))
            issues+=("Service $service is not running")
        fi
    done
    
    # Check resource usage
    local memory_usage=$(free | awk 'FNR==2{printf "%.0f", $3/$2*100}')
    if [ "$memory_usage" -gt 90 ]; then
        health_score=$((health_score - 10))
        issues+=("High memory usage: ${memory_usage}%")
    fi
    
    local disk_usage=$(df / | awk 'FNR==2{print $5}' | sed 's/%//')
    if [ "$disk_usage" -gt 90 ]; then
        health_score=$((health_score - 10))
        issues+=("High disk usage: ${disk_usage}%")
    fi
    
    # API check
    if ! curl -s -f http://localhost/api/health >/dev/null 2>&1; then
        health_score=$((health_score - 20))
        issues+=("API endpoint not responding")
    fi
    
    echo "================================================================"
    echo "                        HEALTH SUMMARY"
    echo "================================================================"
    
    if [ "$health_score" -ge 90 ]; then
        echo -e "Overall Health: ${GREEN}EXCELLENT${NC} ($health_score/100)"
    elif [ "$health_score" -ge 70 ]; then
        echo -e "Overall Health: ${YELLOW}GOOD${NC} ($health_score/100)"
    elif [ "$health_score" -ge 50 ]; then
        echo -e "Overall Health: ${YELLOW}WARNING${NC} ($health_score/100)"
    else
        echo -e "Overall Health: ${RED}CRITICAL${NC} ($health_score/100)"
    fi
    
    if [ ${#issues[@]} -gt 0 ]; then
        echo
        echo "Issues detected:"
        for issue in "${issues[@]}"; do
            echo -e "  ${RED}✗${NC} $issue"
        done
    else
        echo -e "${GREEN}✓ No critical issues detected${NC}"
    fi
    
    echo "================================================================"
}

# Real-time monitoring
monitor_realtime() {
    local interval="${1:-10}"
    
    log "Starting real-time monitoring (refresh every ${interval}s)..."
    log "Press Ctrl+C to stop"
    
    while true; do
        clear
        echo "AUJ Platform Real-Time Monitor - $(date)"
        echo "================================================================"
        
        get_service_status
        get_resource_usage
        
        echo "--- Recent Activity ---"
        journalctl -u auj-platform --since "1 minute ago" --no-pager | tail -3 || echo "No recent activity"
        
        sleep "$interval"
    done
}

# Generate monitoring report
generate_report() {
    local output_file="${1:-auj_monitoring_report_$(date +%Y%m%d_%H%M%S).txt}"
    
    log "Generating monitoring report: $output_file"
    
    {
        echo "AUJ Platform Monitoring Report"
        echo "Generated: $(date)"
        echo "================================================================"
        echo
        
        comprehensive_health_check
        
    } > "$output_file"
    
    log "Report saved to: $output_file"
}

# Alert check
check_alerts() {
    local alerts=()
    
    # Memory check
    local memory_usage=$(free | awk 'FNR==2{printf "%.0f", $3/$2*100}')
    if [ "$memory_usage" -gt 85 ]; then
        alerts+=("MEMORY: High usage ${memory_usage}%")
    fi
    
    # Disk check
    local disk_usage=$(df / | awk 'FNR==2{print $5}' | sed 's/%//')
    if [ "$disk_usage" -gt 90 ]; then
        alerts+=("DISK: High usage ${disk_usage}%")
    fi
    
    # Service check
    for service in "${AUJ_SERVICES[@]}"; do
        if ! systemctl is-active --quiet "$service"; then
            alerts+=("SERVICE: $service is down")
        fi
    done
    
    # Load check
    local load_1m=$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | xargs)
    if (( $(echo "$load_1m > 4.0" | bc -l) )); then
        alerts+=("LOAD: High load average $load_1m")
    fi
    
    if [ ${#alerts[@]} -gt 0 ]; then
        warn "ALERTS DETECTED:"
        for alert in "${alerts[@]}"; do
            echo -e "  ${RED}!${NC} $alert"
        done
        return 1
    else
        log "No alerts detected"
        return 0
    fi
}

# Show help
show_help() {
    cat << EOF
AUJ Platform Monitoring and Diagnostics Script

Usage: $0 COMMAND [OPTIONS]

Commands:
  status                      Show comprehensive system status
  health                      Perform comprehensive health check
  monitor [INTERVAL]          Real-time monitoring (default: 10s)
  report [FILENAME]           Generate monitoring report
  alerts                      Check for system alerts
  
  services                    Show detailed service status
  resources                   Show resource usage
  network                     Check network connectivity
  database                    Check database status
  config                      Validate configuration
  logs [HOURS]               Analyze logs (default: 1 hour)
  performance                 Show performance metrics
  security                    Check security status
  
  help                        Show this help

Examples:
  $0 health                   # Full health check
  $0 monitor 5                # Real-time monitoring every 5 seconds
  $0 logs 24                  # Analyze logs from last 24 hours
  $0 report status.txt        # Generate report to file
  $0 alerts                   # Check for system alerts

Output files:
  Reports: ./auj_monitoring_report_TIMESTAMP.txt
  Logs: $AUJ_LOG_DIR/
  Config: $AUJ_CONFIG_DIR/
EOF
}

# Main function
main() {
    case "${1:-status}" in
        "status"|"health")
            comprehensive_health_check
            ;;
        "monitor")
            monitor_realtime "$2"
            ;;
        "report")
            generate_report "$2"
            ;;
        "alerts")
            check_alerts
            ;;
        "services")
            get_service_status
            ;;
        "resources")
            get_resource_usage
            ;;
        "network")
            check_network
            ;;
        "database")
            check_database
            ;;
        "config")
            check_configuration
            ;;
        "logs")
            analyze_logs "$2"
            ;;
        "performance")
            get_performance_metrics
            ;;
        "security")
            security_check
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run main function
main "$@"