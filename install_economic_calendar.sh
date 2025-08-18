#!/bin/bash

# AUJ Platform Economic Calendar - Quick Install Script
# This script automates the installation and setup of the economic calendar system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="/opt/auj-platform"
PYTHON_VERSION="3.11"
SERVICE_USER="auj"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command_exists apt; then
            echo "ubuntu"
        elif command_exists yum; then
            echo "centos"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

# Function to install system dependencies
install_system_deps() {
    local os=$(detect_os)
    print_status "Installing system dependencies for $os..."
    
    case $os in
        "ubuntu")
            sudo apt update
            sudo apt install -y python3.11 python3.11-venv python3-pip postgresql-client git curl wget
            ;;
        "centos")
            sudo yum update -y
            sudo yum install -y python3.11 python3-pip postgresql git curl wget
            ;;
        "macos")
            if ! command_exists brew; then
                print_status "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            brew install python@3.11 postgresql git
            ;;
        *)
            print_error "Unsupported operating system: $os"
            exit 1
            ;;
    esac
    
    print_success "System dependencies installed"
}

# Function to create service user
create_service_user() {
    if id "$SERVICE_USER" &>/dev/null; then
        print_warning "User $SERVICE_USER already exists"
    else
        print_status "Creating service user: $SERVICE_USER"
        sudo useradd -r -s /bin/bash -m -d /home/$SERVICE_USER $SERVICE_USER
        print_success "Service user created"
    fi
}

# Function to setup installation directory
setup_install_dir() {
    print_status "Setting up installation directory: $INSTALL_DIR"
    
    if [[ -d "$INSTALL_DIR" ]]; then
        print_warning "Installation directory already exists"
        read -p "Do you want to continue? This will overwrite existing files. (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_error "Installation cancelled"
            exit 1
        fi
    fi
    
    sudo mkdir -p "$INSTALL_DIR"
    sudo chown "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
    print_success "Installation directory ready"
}

# Function to clone repository
clone_repository() {
    print_status "Cloning AUJ Platform repository..."
    
    if [[ -d "$INSTALL_DIR/.git" ]]; then
        print_status "Repository already exists, pulling latest changes..."
        cd "$INSTALL_DIR"
        sudo -u "$SERVICE_USER" git pull
    else
        # For this demo, we'll copy the current directory
        print_status "Copying current directory to installation location..."
        sudo cp -r . "$INSTALL_DIR/"
        sudo chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
    fi
    
    print_success "Repository ready"
}

# Function to setup Python virtual environment
setup_python_env() {
    print_status "Setting up Python virtual environment..."
    
    cd "$INSTALL_DIR"
    sudo -u "$SERVICE_USER" python3.11 -m venv venv
    sudo -u "$SERVICE_USER" venv/bin/pip install --upgrade pip
    
    # Install requirements
    if [[ -f "auj_platform/requirements.txt" ]]; then
        sudo -u "$SERVICE_USER" venv/bin/pip install -r auj_platform/requirements.txt
    fi
    
    # Install additional economic calendar dependencies
    sudo -u "$SERVICE_USER" venv/bin/pip install beautifulsoup4 lxml requests-html selenium webdriver-manager schedule
    
    print_success "Python environment ready"
}

# Function to setup database
setup_database() {
    print_status "Setting up database..."
    
    # Check if PostgreSQL is running
    if systemctl is-active --quiet postgresql; then
        print_status "PostgreSQL is running"
        
        # Create database and user
        sudo -u postgres psql -c "CREATE DATABASE auj_platform;" 2>/dev/null || print_warning "Database auj_platform already exists"
        sudo -u postgres psql -c "CREATE USER auj_user WITH PASSWORD 'auj_secure_password';" 2>/dev/null || print_warning "User auj_user already exists"
        sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE auj_platform TO auj_user;"
        
        # Run migration
        cd "$INSTALL_DIR/auj_platform"
        sudo -u "$SERVICE_USER" ../venv/bin/python src/database/migrate_economic_calendar_db.py
        
        print_success "Database setup completed"
    else
        print_warning "PostgreSQL not running, using SQLite instead"
        
        # Create SQLite database
        cd "$INSTALL_DIR/auj_platform"
        sudo -u "$SERVICE_USER" mkdir -p data
        sudo -u "$SERVICE_USER" ../venv/bin/python src/database/migrate_economic_calendar_db.py
        
        print_success "SQLite database setup completed"
    fi
}

# Function to create configuration
create_configuration() {
    print_status "Creating configuration files..."
    
    cd "$INSTALL_DIR"
    
    # Create environment file
    sudo -u "$SERVICE_USER" tee .env > /dev/null << EOF
# AUJ Platform Environment Configuration
AUJ_DB_HOST=localhost
AUJ_DB_PORT=5432
AUJ_DB_NAME=auj_platform
AUJ_DB_USER=auj_user
AUJ_DB_PASS=auj_secure_password

# Economic Calendar API Keys (Optional)
AUJ_TRADINGECONOMICS_API_KEY=
AUJ_CUSTOM_ECONOMIC_API_KEY=

# RabbitMQ Configuration
AUJ_RABBITMQ_HOST=localhost
AUJ_RABBITMQ_USER=auj_user
AUJ_RABBITMQ_PASS=auj_rabbitmq_password

# Security
AUJ_ENCRYPTION_KEY=$(openssl rand -base64 32)
AUJ_JWT_SECRET=$(openssl rand -base64 32)
EOF
    
    print_success "Configuration files created"
}

# Function to create systemd services
create_systemd_services() {
    print_status "Creating systemd services..."
    
    # Economic Monitor Service
    sudo tee /etc/systemd/system/auj-economic-monitor.service > /dev/null << EOF
[Unit]
Description=AUJ Economic Calendar Monitor
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/python auj_platform/src/monitoring/economic_monitor.py --start
Restart=always
RestartSec=10
Environment=PYTHONPATH=$INSTALL_DIR/auj_platform/src

[Install]
WantedBy=multi-user.target
EOF

    # Dashboard Service
    sudo tee /etc/systemd/system/auj-dashboard.service > /dev/null << EOF
[Unit]
Description=AUJ Platform Dashboard
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR/auj_platform/dashboard
ExecStart=$INSTALL_DIR/venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10
Environment=PYTHONPATH=$INSTALL_DIR/auj_platform/src

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd
    sudo systemctl daemon-reload
    
    print_success "Systemd services created"
}

# Function to run tests
run_tests() {
    print_status "Running system tests..."
    
    cd "$INSTALL_DIR"
    
    # Test database connection
    sudo -u "$SERVICE_USER" venv/bin/python -c "
import sys
sys.path.append('auj_platform/src')
from database import DatabaseManager
db = DatabaseManager()
print('✅ Database connection successful')
"
    
    # Test economic providers
    sudo -u "$SERVICE_USER" venv/bin/python -c "
import sys
sys.path.append('auj_platform/src')
from analytics.providers.economic_calendar_providers import ForexFactoryProvider
ff = ForexFactoryProvider()
print('✅ Economic providers initialized')
"
    
    # Test indicators
    sudo -u "$SERVICE_USER" venv/bin/python -c "
import sys
sys.path.append('auj_platform/src')
from analytics.indicators.economic.economic_event_impact_indicator import EconomicEventImpactIndicator
indicator = EconomicEventImpactIndicator()
print('✅ Economic indicators initialized')
"
    
    print_success "All tests passed"
}

# Function to start services
start_services() {
    print_status "Starting services..."
    
    # Enable and start services
    sudo systemctl enable auj-economic-monitor
    sudo systemctl enable auj-dashboard
    
    sudo systemctl start auj-economic-monitor
    sudo systemctl start auj-dashboard
    
    # Wait a moment for services to start
    sleep 5
    
    # Check service status
    if systemctl is-active --quiet auj-economic-monitor; then
        print_success "Economic monitor service started"
    else
        print_error "Economic monitor service failed to start"
        sudo systemctl status auj-economic-monitor --no-pager
    fi
    
    if systemctl is-active --quiet auj-dashboard; then
        print_success "Dashboard service started"
    else
        print_error "Dashboard service failed to start"
        sudo systemctl status auj-dashboard --no-pager
    fi
}

# Function to display final information
display_final_info() {
    print_success "AUJ Platform Economic Calendar installation completed!"
    echo
    echo "🚀 Services Status:"
    echo "   📊 Dashboard: http://localhost:8501"
    echo "   ⚡ Economic Monitor: Active in background"
    echo
    echo "📋 Management Commands:"
    echo "   sudo systemctl status auj-dashboard"
    echo "   sudo systemctl status auj-economic-monitor"
    echo "   sudo systemctl restart auj-dashboard"
    echo "   sudo systemctl restart auj-economic-monitor"
    echo
    echo "📁 Installation Directory: $INSTALL_DIR"
    echo "👤 Service User: $SERVICE_USER"
    echo
    echo "📖 Full Documentation: $INSTALL_DIR/docs/economic_calendar_deployment_guide.md"
    echo
    echo "🎯 Quick Access:"
    echo "   Dashboard: Navigate to '📅 Economic Calendar' tab"
    echo "   Logs: tail -f $INSTALL_DIR/auj_platform/logs/auj_platform.log"
    echo
}

# Main installation function
main() {
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                   AUJ Platform Economic Calendar              ║"
    echo "║                        Quick Install Script                    ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        print_error "This script should not be run as root. Please run as a regular user with sudo privileges."
        exit 1
    fi
    
    # Check for sudo privileges
    if ! sudo -v; then
        print_error "This script requires sudo privileges"
        exit 1
    fi
    
    print_status "Starting AUJ Platform Economic Calendar installation..."
    
    # Installation steps
    install_system_deps
    create_service_user
    setup_install_dir
    clone_repository
    setup_python_env
    setup_database
    create_configuration
    create_systemd_services
    run_tests
    start_services
    
    display_final_info
}

# Script options
case "${1:-}" in
    "uninstall")
        print_status "Uninstalling AUJ Platform..."
        sudo systemctl stop auj-dashboard auj-economic-monitor 2>/dev/null || true
        sudo systemctl disable auj-dashboard auj-economic-monitor 2>/dev/null || true
        sudo rm -f /etc/systemd/system/auj-*.service
        sudo systemctl daemon-reload
        sudo rm -rf "$INSTALL_DIR"
        sudo userdel -r "$SERVICE_USER" 2>/dev/null || true
        print_success "Uninstallation completed"
        ;;
    "status")
        echo "Service Status:"
        sudo systemctl status auj-dashboard --no-pager
        sudo systemctl status auj-economic-monitor --no-pager
        ;;
    "restart")
        print_status "Restarting services..."
        sudo systemctl restart auj-dashboard auj-economic-monitor
        print_success "Services restarted"
        ;;
    "logs")
        echo "Recent logs:"
        sudo journalctl -u auj-dashboard -u auj-economic-monitor --no-pager -n 50
        ;;
    "help"|"--help"|"-h")
        echo "AUJ Platform Economic Calendar Install Script"
        echo
        echo "Usage: $0 [OPTION]"
        echo
        echo "Options:"
        echo "  (no option)    Install AUJ Platform Economic Calendar"
        echo "  uninstall      Remove AUJ Platform installation"
        echo "  status         Show service status"
        echo "  restart        Restart services"
        echo "  logs           Show recent logs"
        echo "  help           Show this help"
        ;;
    *)
        main
        ;;
esac