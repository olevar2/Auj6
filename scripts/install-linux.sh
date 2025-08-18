#!/bin/bash
# AUJ Platform Linux Installation Script
# Automated installation and setup for Linux deployment with MetaApi

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
AUJ_USER="auj"
AUJ_GROUP="auj"
AUJ_HOME="/opt/auj-platform"
AUJ_VENV="${AUJ_HOME}/venv"
AUJ_LOG_DIR="/var/log/auj-platform"
AUJ_CONFIG_DIR="/etc/auj-platform"
AUJ_DATA_DIR="${AUJ_HOME}/data"
PYTHON_VERSION="3.11"

# Logging function
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

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root. Use: sudo $0"
    fi
}

# Detect Linux distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        VERSION=$VERSION_ID
    else
        error "Cannot detect Linux distribution"
    fi
    
    log "Detected distribution: $DISTRO $VERSION"
}

# Install system dependencies
install_system_deps() {
    log "Installing system dependencies..."
    
    case $DISTRO in
        ubuntu|debian)
            apt-get update
            apt-get install -y \
                python${PYTHON_VERSION} \
                python${PYTHON_VERSION}-dev \
                python${PYTHON_VERSION}-venv \
                python3-pip \
                build-essential \
                libssl-dev \
                libffi-dev \
                libpq-dev \
                curl \
                wget \
                git \
                htop \
                supervisor \
                nginx \
                certbot \
                python3-certbot-nginx \
                redis-server \
                sqlite3 \
                jq \
                unzip \
                systemd \
                logrotate
            ;;
        centos|rhel|fedora)
            if command -v dnf &> /dev/null; then
                PKG_MANAGER="dnf"
            else
                PKG_MANAGER="yum"
            fi
            
            $PKG_MANAGER update -y
            $PKG_MANAGER install -y \
                python${PYTHON_VERSION} \
                python${PYTHON_VERSION}-devel \
                python3-pip \
                gcc \
                gcc-c++ \
                make \
                openssl-devel \
                libffi-devel \
                postgresql-devel \
                curl \
                wget \
                git \
                htop \
                supervisor \
                nginx \
                certbot \
                python3-certbot-nginx \
                redis \
                sqlite \
                jq \
                unzip \
                systemd \
                logrotate
            ;;
        alpine)
            apk update
            apk add \
                python3 \
                python3-dev \
                py3-pip \
                build-base \
                openssl-dev \
                libffi-dev \
                postgresql-dev \
                curl \
                wget \
                git \
                htop \
                supervisor \
                nginx \
                certbot \
                redis \
                sqlite \
                jq \
                unzip \
                logrotate
            ;;
        *)
            error "Unsupported distribution: $DISTRO"
            ;;
    esac
    
    log "System dependencies installed successfully"
}

# Create user and directories
setup_user_and_dirs() {
    log "Setting up user and directories..."
    
    # Create auj user if it doesn't exist
    if ! id -u $AUJ_USER >/dev/null 2>&1; then
        useradd -r -m -s /bin/bash -d $AUJ_HOME $AUJ_USER
        usermod -a -G sudo $AUJ_USER
        log "Created user: $AUJ_USER"
    else
        info "User $AUJ_USER already exists"
    fi
    
    # Create directories
    mkdir -p $AUJ_HOME
    mkdir -p $AUJ_LOG_DIR
    mkdir -p $AUJ_CONFIG_DIR
    mkdir -p $AUJ_DATA_DIR
    mkdir -p $AUJ_HOME/backups
    mkdir -p $AUJ_HOME/scripts
    mkdir -p $AUJ_HOME/services
    
    # Set ownership and permissions
    chown -R $AUJ_USER:$AUJ_GROUP $AUJ_HOME
    chown -R $AUJ_USER:$AUJ_GROUP $AUJ_LOG_DIR
    chown -R root:$AUJ_GROUP $AUJ_CONFIG_DIR
    
    chmod 755 $AUJ_HOME
    chmod 755 $AUJ_LOG_DIR
    chmod 750 $AUJ_CONFIG_DIR
    chmod 700 $AUJ_DATA_DIR
    
    log "User and directories setup completed"
}

# Install Python dependencies
install_python_deps() {
    log "Setting up Python virtual environment..."
    
    # Create virtual environment
    sudo -u $AUJ_USER python${PYTHON_VERSION} -m venv $AUJ_VENV
    
    # Activate virtual environment and upgrade pip
    sudo -u $AUJ_USER $AUJ_VENV/bin/pip install --upgrade pip setuptools wheel
    
    # Install core dependencies
    sudo -u $AUJ_USER $AUJ_VENV/bin/pip install \
        metaapi-cloud-sdk>=27.0.0 \
        websockets>=11.0.0 \
        aiohttp>=3.8.0 \
        asyncio-mqtt>=0.13.0 \
        pandas>=2.0.0 \
        numpy>=1.24.0 \
        scipy>=1.10.0 \
        scikit-learn>=1.3.0 \
        fastapi>=0.104.0 \
        uvicorn>=0.24.0 \
        sqlalchemy>=2.0.0 \
        alembic>=1.12.0 \
        redis>=5.0.0 \
        celery>=5.3.0 \
        pyyaml>=6.0 \
        python-dotenv>=1.0.0 \
        psutil>=5.9.0 \
        prometheus-client>=0.19.0 \
        streamlit>=1.28.0
    
    # Install requirements if file exists
    if [ -f "requirements-linux.txt" ]; then
        sudo -u $AUJ_USER $AUJ_VENV/bin/pip install -r requirements-linux.txt
        log "Installed requirements from requirements-linux.txt"
    fi
    
    log "Python dependencies installed successfully"
}

# Setup configuration
setup_config() {
    log "Setting up configuration files..."
    
    # Copy configuration files
    if [ -d "config" ]; then
        cp -r config/* $AUJ_CONFIG_DIR/
        chown -R root:$AUJ_GROUP $AUJ_CONFIG_DIR
        chmod -R 640 $AUJ_CONFIG_DIR/*.yaml
        chmod 600 $AUJ_CONFIG_DIR/.env.template
        
        log "Configuration files copied to $AUJ_CONFIG_DIR"
    else
        warn "Config directory not found, skipping configuration setup"
    fi
    
    # Create .env file from template if it doesn't exist
    if [ -f "$AUJ_CONFIG_DIR/.env.template" ] && [ ! -f "$AUJ_HOME/.env" ]; then
        cp $AUJ_CONFIG_DIR/.env.template $AUJ_HOME/.env
        chown $AUJ_USER:$AUJ_GROUP $AUJ_HOME/.env
        chmod 600 $AUJ_HOME/.env
        
        warn "Created .env file from template. Please edit $AUJ_HOME/.env with your actual values."
    fi
}

# Setup systemd services
setup_systemd() {
    log "Setting up systemd services..."
    
    # AUJ Platform main service
    cat > /etc/systemd/system/auj-platform.service << EOF
[Unit]
Description=AUJ Trading Platform
After=network-online.target redis.service
Wants=network-online.target
Requires=redis.service

[Service]
Type=notify
User=$AUJ_USER
Group=$AUJ_GROUP
WorkingDirectory=$AUJ_HOME
Environment=PATH=$AUJ_VENV/bin
EnvironmentFile=$AUJ_HOME/.env
ExecStart=$AUJ_VENV/bin/python -m auj_platform.main
ExecReload=/bin/kill -HUP \$MAINPID
KillMode=mixed
TimeoutStopSec=60
Restart=always
RestartSec=10

# Resource limits
MemoryLimit=4G
CPUQuota=200%

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$AUJ_HOME $AUJ_LOG_DIR /tmp
PrivateTmp=true

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=auj-platform

[Install]
WantedBy=multi-user.target
EOF

    # AUJ MetaApi service
    cat > /etc/systemd/system/auj-metaapi.service << EOF
[Unit]
Description=AUJ MetaApi Integration Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$AUJ_USER
Group=$AUJ_GROUP
WorkingDirectory=$AUJ_HOME
Environment=PATH=$AUJ_VENV/bin
EnvironmentFile=$AUJ_HOME/.env
ExecStart=$AUJ_VENV/bin/python -m auj_platform.services.metaapi_service
Restart=always
RestartSec=15

# Resource limits
MemoryLimit=1G
CPUQuota=100%

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$AUJ_HOME $AUJ_LOG_DIR /tmp
PrivateTmp=true

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=auj-metaapi

[Install]
WantedBy=multi-user.target
EOF

    # Dashboard service
    cat > /etc/systemd/system/auj-dashboard.service << EOF
[Unit]
Description=AUJ Platform Dashboard
After=auj-platform.service
Requires=auj-platform.service

[Service]
Type=simple
User=$AUJ_USER
Group=$AUJ_GROUP
WorkingDirectory=$AUJ_HOME
Environment=PATH=$AUJ_VENV/bin
EnvironmentFile=$AUJ_HOME/.env
ExecStart=$AUJ_VENV/bin/streamlit run auj_platform/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10

# Resource limits
MemoryLimit=2G
CPUQuota=100%

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=auj-dashboard

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd and enable services
    systemctl daemon-reload
    systemctl enable auj-platform.service
    systemctl enable auj-metaapi.service
    systemctl enable auj-dashboard.service
    systemctl enable redis.service
    
    log "Systemd services configured and enabled"
}

# Setup nginx reverse proxy
setup_nginx() {
    log "Setting up Nginx reverse proxy..."
    
    cat > /etc/nginx/sites-available/auj-platform << EOF
server {
    listen 80;
    server_name localhost auj-platform.local;

    # API proxy
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Dashboard proxy
    location / {
        proxy_pass http://127.0.0.1:8501/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket support for Streamlit
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Streamlit specific headers
        proxy_set_header X-Forwarded-Host \$host;
        proxy_set_header X-Forwarded-Server \$host;
        proxy_buffering off;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }

    # Static files
    location /static/ {
        alias $AUJ_HOME/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF

    # Enable site
    ln -sf /etc/nginx/sites-available/auj-platform /etc/nginx/sites-enabled/
    
    # Remove default site
    rm -f /etc/nginx/sites-enabled/default
    
    # Test nginx configuration
    nginx -t
    
    # Enable and start nginx
    systemctl enable nginx
    systemctl restart nginx
    
    log "Nginx reverse proxy configured"
}

# Setup log rotation
setup_logrotate() {
    log "Setting up log rotation..."
    
    cat > /etc/logrotate.d/auj-platform << EOF
$AUJ_LOG_DIR/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $AUJ_USER $AUJ_GROUP
    postrotate
        systemctl reload auj-platform || true
        systemctl reload auj-metaapi || true
        systemctl reload auj-dashboard || true
    endscript
}
EOF

    log "Log rotation configured"
}

# Setup firewall
setup_firewall() {
    log "Setting up firewall..."
    
    if command -v ufw &> /dev/null; then
        # Ubuntu/Debian UFW
        ufw --force enable
        ufw default deny incoming
        ufw default allow outgoing
        
        # SSH
        ufw allow ssh
        
        # HTTP/HTTPS
        ufw allow 80/tcp
        ufw allow 443/tcp
        
        # Allow internal communication
        ufw allow from 10.0.0.0/8 to any port 8000
        ufw allow from 10.0.0.0/8 to any port 8501
        
        log "UFW firewall configured"
        
    elif command -v firewall-cmd &> /dev/null; then
        # CentOS/RHEL/Fedora firewalld
        systemctl enable firewalld
        systemctl start firewalld
        
        firewall-cmd --permanent --add-service=ssh
        firewall-cmd --permanent --add-service=http
        firewall-cmd --permanent --add-service=https
        
        # Internal ports
        firewall-cmd --permanent --add-rich-rule="rule family='ipv4' source address='10.0.0.0/8' port protocol='tcp' port='8000' accept"
        firewall-cmd --permanent --add-rich-rule="rule family='ipv4' source address='10.0.0.0/8' port protocol='tcp' port='8501' accept"
        
        firewall-cmd --reload
        
        log "Firewalld configured"
    else
        warn "No supported firewall found. Please configure manually."
    fi
}

# Performance tuning
tune_system() {
    log "Applying system performance tuning..."
    
    # Kernel parameters
    cat > /etc/sysctl.d/99-auj-platform.conf << EOF
# Network optimization
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_fin_timeout = 30
net.ipv4.tcp_keepalive_time = 1200
net.ipv4.tcp_keepalive_probes = 3
net.ipv4.tcp_keepalive_intvl = 30

# Memory management
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5

# File system
fs.file-max = 100000
fs.inotify.max_user_watches = 524288
EOF

    # Apply sysctl settings
    sysctl -p /etc/sysctl.d/99-auj-platform.conf
    
    # System limits
    cat > /etc/security/limits.d/auj-platform.conf << EOF
# AUJ Platform limits
$AUJ_USER soft nofile 65535
$AUJ_USER hard nofile 65535
$AUJ_USER soft nproc 32768
$AUJ_USER hard nproc 32768
$AUJ_USER soft memlock unlimited
$AUJ_USER hard memlock unlimited
EOF

    log "System performance tuning applied"
}

# Create management scripts
create_management_scripts() {
    log "Creating management scripts..."
    
    # Status script
    cat > $AUJ_HOME/scripts/status.sh << 'EOF'
#!/bin/bash
# AUJ Platform Status Script

echo "=== AUJ Platform Status ==="
echo

echo "Services Status:"
systemctl status auj-platform --no-pager -l
systemctl status auj-metaapi --no-pager -l
systemctl status auj-dashboard --no-pager -l
systemctl status redis --no-pager -l
systemctl status nginx --no-pager -l

echo
echo "System Resources:"
free -h
df -h
uptime

echo
echo "Recent Logs:"
journalctl -u auj-platform --since "1 hour ago" --no-pager | tail -10
EOF

    # Start script
    cat > $AUJ_HOME/scripts/start.sh << 'EOF'
#!/bin/bash
# AUJ Platform Start Script

echo "Starting AUJ Platform services..."

systemctl start redis
systemctl start auj-platform
systemctl start auj-metaapi
systemctl start auj-dashboard
systemctl start nginx

echo "All services started. Check status with: ./status.sh"
EOF

    # Stop script
    cat > $AUJ_HOME/scripts/stop.sh << 'EOF'
#!/bin/bash
# AUJ Platform Stop Script

echo "Stopping AUJ Platform services..."

systemctl stop auj-dashboard
systemctl stop auj-metaapi
systemctl stop auj-platform
systemctl stop nginx

echo "All services stopped."
EOF

    # Backup script
    cat > $AUJ_HOME/scripts/backup.sh << 'EOF'
#!/bin/bash
# AUJ Platform Backup Script

BACKUP_DIR="/opt/auj-platform/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="auj_backup_$DATE.tar.gz"

echo "Creating backup: $BACKUP_FILE"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database, config, and logs
tar -czf $BACKUP_DIR/$BACKUP_FILE \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='venv' \
    /opt/auj-platform/data \
    /etc/auj-platform \
    /var/log/auj-platform

echo "Backup created: $BACKUP_DIR/$BACKUP_FILE"

# Keep only last 7 backups
find $BACKUP_DIR -name "auj_backup_*.tar.gz" -mtime +7 -delete

echo "Old backups cleaned up."
EOF

    # Make scripts executable
    chmod +x $AUJ_HOME/scripts/*.sh
    chown -R $AUJ_USER:$AUJ_GROUP $AUJ_HOME/scripts
    
    log "Management scripts created"
}

# Final verification
verify_installation() {
    log "Verifying installation..."
    
    # Check services
    if systemctl is-enabled auj-platform >/dev/null 2>&1; then
        info "✓ AUJ Platform service enabled"
    else
        warn "✗ AUJ Platform service not enabled"
    fi
    
    if systemctl is-enabled nginx >/dev/null 2>&1; then
        info "✓ Nginx service enabled"
    else
        warn "✗ Nginx service not enabled"
    fi
    
    # Check Python environment
    if [ -f "$AUJ_VENV/bin/python" ]; then
        info "✓ Python virtual environment created"
        PYTHON_VER=$($AUJ_VENV/bin/python --version)
        info "  Python version: $PYTHON_VER"
    else
        warn "✗ Python virtual environment not found"
    fi
    
    # Check configuration
    if [ -f "$AUJ_HOME/.env" ]; then
        info "✓ Environment file exists"
        warn "  Please edit $AUJ_HOME/.env with your actual configuration"
    else
        warn "✗ Environment file not found"
    fi
    
    log "Installation verification completed"
}

# Display post-installation instructions
show_instructions() {
    echo
    echo "================================================================"
    echo "           AUJ Platform Installation Complete!"
    echo "================================================================"
    echo
    echo "Next steps:"
    echo "1. Edit configuration file: $AUJ_HOME/.env"
    echo "   - Set your MetaApi token and account ID"
    echo "   - Configure other settings as needed"
    echo
    echo "2. Start services:"
    echo "   sudo $AUJ_HOME/scripts/start.sh"
    echo
    echo "3. Check status:"
    echo "   sudo $AUJ_HOME/scripts/status.sh"
    echo
    echo "4. Access the platform:"
    echo "   - Dashboard: http://localhost (or your server IP)"
    echo "   - API: http://localhost/api"
    echo
    echo "5. View logs:"
    echo "   journalctl -u auj-platform -f"
    echo
    echo "Configuration files:"
    echo "- Main config: $AUJ_CONFIG_DIR/"
    echo "- Environment: $AUJ_HOME/.env"
    echo "- Data: $AUJ_DATA_DIR/"
    echo "- Logs: $AUJ_LOG_DIR/"
    echo
    echo "Management scripts:"
    echo "- Start: $AUJ_HOME/scripts/start.sh"
    echo "- Stop: $AUJ_HOME/scripts/stop.sh"
    echo "- Status: $AUJ_HOME/scripts/status.sh"
    echo "- Backup: $AUJ_HOME/scripts/backup.sh"
    echo
    echo "================================================================"
}

# Main installation function
main() {
    log "Starting AUJ Platform Linux installation..."
    
    check_root
    detect_distro
    install_system_deps
    setup_user_and_dirs
    install_python_deps
    setup_config
    setup_systemd
    setup_nginx
    setup_logrotate
    setup_firewall
    tune_system
    create_management_scripts
    verify_installation
    show_instructions
    
    log "Installation completed successfully!"
}

# Run main function
main "$@"