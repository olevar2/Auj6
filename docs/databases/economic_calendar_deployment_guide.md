# AUJ Platform Economic Calendar - Complete Deployment Guide

## 📚 Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Quick Start Guide](#quick-start-guide)
4. [Detailed Installation](#detailed-installation)
5. [Configuration](#configuration)
6. [Testing & Validation](#testing--validation)
7. [Production Deployment](#production-deployment)
8. [Monitoring & Maintenance](#monitoring--maintenance)
9. [Troubleshooting](#troubleshooting)
10. [API Reference](#api-reference)

---

## 🎯 Overview

The AUJ Platform Economic Calendar is a comprehensive system that integrates real-time economic event monitoring with automated trading signal generation. The system consists of:

### 🏗️ Core Components

- **📊 Data Providers**: ForexFactory, Investing.com, TradingEconomics
- **🧠 Economic Indicators**: 5 specialized indicators for event analysis
- **🤖 Enhanced Agents**: 4 trading agents with economic calendar integration
- **🗄️ Database Schema**: 6 tables for comprehensive economic data storage
- **📈 Dashboard Integration**: Streamlit-based economic calendar interface
- **⚡ Real-time Monitoring**: Background service for continuous event monitoring
- **📱 Alert System**: Real-time notifications for high-impact events

### 🚀 Key Features

- ✅ **Multi-provider data aggregation** from 3 major economic calendar sources
- ✅ **Real-time event monitoring** with automatic signal generation
- ✅ **Advanced economic indicators** for impact analysis and prediction
- ✅ **Smart agent integration** with economic context awareness
- ✅ **Comprehensive dashboard** with interactive economic calendar
- ✅ **Performance tracking** with detailed analytics and correlations
- ✅ **Production-ready architecture** with Docker support and monitoring

---

## 💻 System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+), Windows 10+, macOS 10.15+
- **RAM**: 4GB (8GB recommended)
- **Storage**: 10GB free space
- **Python**: 3.9+ (3.11 recommended)
- **Network**: Stable internet connection for data providers

### Recommended Production Setup
- **OS**: Ubuntu 22.04 LTS
- **RAM**: 16GB+ 
- **CPU**: 4+ cores
- **Storage**: 50GB+ SSD
- **Database**: PostgreSQL 15+
- **Message Broker**: RabbitMQ 3.8+
- **Monitoring**: Prometheus + Grafana

### Dependencies
```bash
# Core Python packages
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.15.0
requests>=2.31.0
beautifulsoup4>=4.12.0
sqlalchemy>=2.0.0
schedule>=1.2.0

# Database drivers
psycopg2-binary>=2.9.0  # PostgreSQL
sqlite3  # Built-in with Python

# Message broker
pika>=1.3.0  # RabbitMQ

# Testing framework
pytest>=7.4.0
pytest-asyncio>=0.21.0
```

---

## ⚡ Quick Start Guide

### 1. Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd AUJ

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize Database
```bash
# Navigate to project root
cd auj_platform

# Run database migration
python src/database/migrate_economic_calendar_db.py

# Verify tables created
python -c "
from src.database import DatabaseManager
db = DatabaseManager()
print('Database initialized successfully!')
print('Tables:', db.get_table_names())
"
```

### 3. Test Economic Calendar
```bash
# Run comprehensive tests
cd tests
python test_economic_calendar_comprehensive.py

# Expected output: All tests pass
```

### 4. Launch Dashboard
```bash
# Start the dashboard
cd auj_platform/dashboard
streamlit run app.py

# Open browser to: http://localhost:8501
# Navigate to: 📅 Economic Calendar tab
```

### 5. Start Monitoring (Optional)
```bash
# Start real-time monitoring
cd auj_platform/src
python monitoring/economic_monitor.py --start

# Check status
python monitoring/economic_monitor.py --status
```

**🎉 You're ready to go!** The economic calendar should be fully functional with live data from economic providers.

---

## 🔧 Detailed Installation

### Step 1: Environment Setup

#### Linux/Ubuntu
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3.11 python3.11-venv python3-pip postgresql-client -y

# Create project directory
mkdir -p /opt/auj-platform
cd /opt/auj-platform

# Clone repository
git clone <repository-url> .
```

#### Windows
```powershell
# Install Python 3.11 from python.org
# Install Git from git-scm.com

# Clone repository
git clone <repository-url>
cd AUJ
```

#### macOS
```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Clone repository
git clone <repository-url>
cd AUJ
```

### Step 2: Virtual Environment
```bash
# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r auj_platform/requirements.txt

# Install additional economic calendar dependencies
pip install beautifulsoup4 lxml requests-html selenium webdriver-manager
```

### Step 3: Database Setup

#### SQLite (Development)
```bash
# SQLite is included by default
# Database will be created automatically at: auj_platform/data/auj_platform.db

# Run migration script
cd auj_platform
python src/database/migrate_economic_calendar_db.py
```

#### PostgreSQL (Production)
```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib  # Ubuntu
brew install postgresql  # macOS

# Create database and user
sudo -u postgres psql
CREATE DATABASE auj_platform;
CREATE USER auj_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE auj_platform TO auj_user;
\q

# Update configuration
export AUJ_DB_USER=auj_user
export AUJ_DB_PASS=secure_password

# Run migration
python src/database/migrate_economic_calendar_db.py --db-url="postgresql://auj_user:secure_password@localhost/auj_platform"
```

### Step 4: Configuration

#### Environment Variables
```bash
# Create .env file
cat > .env << EOF
# Database Configuration
AUJ_DB_USER=auj_user
AUJ_DB_PASS=secure_password
AUJ_DB_HOST=localhost
AUJ_DB_PORT=5432
AUJ_DB_NAME=auj_platform

# Economic Calendar API Keys (Optional)
AUJ_TRADINGECONOMICS_API_KEY=your_api_key_here
AUJ_CUSTOM_ECONOMIC_API_KEY=your_custom_key_here

# RabbitMQ Configuration
AUJ_RABBITMQ_USER=auj_user
AUJ_RABBITMQ_PASSWORD=secure_password

# Security
AUJ_ENCRYPTION_KEY=your_32_character_encryption_key
AUJ_JWT_SECRET=your_jwt_secret_key
EOF

# Load environment variables
source .env
```

#### Main Configuration
```bash
# Copy configuration template
cp config/main_config.yaml.template config/main_config.yaml

# Edit configuration (already optimized for economic calendar)
vim config/main_config.yaml

# Key settings to verify:
# - economic_calendar.enabled: true
# - feature_flags.enable_economic_calendar: true
# - feature_flags.enable_economic_event_signals: true
```

### Step 5: Validation
```bash
# Run system validation
python -c "
import sys
sys.path.append('auj_platform/src')

# Test database connection
from database import DatabaseManager
db = DatabaseManager()
print('✅ Database connection successful')

# Test economic providers
from analytics.providers.economic_calendar_providers import ForexFactoryProvider
ff = ForexFactoryProvider()
print('✅ ForexFactory provider initialized')

# Test indicators
from analytics.indicators.economic.economic_event_impact_indicator import EconomicEventImpactIndicator
indicator = EconomicEventImpactIndicator()
print('✅ Economic indicators initialized')

print('🎉 All components validated successfully!')
"
```

---

## ⚙️ Configuration

### Main Configuration (config/main_config.yaml)

The economic calendar is configured through several key sections:

#### Economic Calendar Settings
```yaml
data_providers:
  economic_calendar:
    enabled: true
    primary_source: "forexfactory"
    backup_sources: ["investing"]
    request_timeout: 30
    cache_duration: 60
    
    # API Keys for premium sources
    api_keys:
      tradingeconomics: "${AUJ_TRADINGECONOMICS_API_KEY}"
      custom: "${AUJ_CUSTOM_ECONOMIC_API_KEY}"
    
    # Default filters
    default_filters:
      currencies: ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
      impact_levels: ["HIGH", "MEDIUM", "CRITICAL"]
      max_events_per_request: 100
```

#### Feature Flags
```yaml
feature_flags:
  enable_economic_calendar: true
  enable_economic_event_signals: true
  enable_event_correlation_analysis: true
  enable_sentiment_analysis: true
  enable_news_integration: true
```

#### Monitoring Configuration
```yaml
monitoring:
  economic_monitor:
    enabled: true
    update_interval_seconds: 60
    alert_thresholds:
      high_impact_score: 0.8
      volatility_threshold: 0.7
      confluence_threshold: 0.6
```

### Dashboard Configuration

The dashboard configuration is automatically detected from `auj_platform/dashboard/dashboard_config.py`:

```python
FEATURE_FLAGS = {
    'economic_calendar': True,
    'economic_event_signals': True,
    'real_time_monitoring': True,
    'advanced_analytics': True
}
```

### Provider-Specific Configuration

#### ForexFactory Provider
```python
# No API key required
# Rate limiting: 1 request per second
# Covers: All major economic events
# Data format: HTML scraping
```

#### Investing.com Provider
```python
# No API key required
# Rate limiting: 2 requests per second
# Covers: Global economic events
# Data format: JSON/HTML hybrid
```

#### TradingEconomics Provider (Premium)
```python
# Requires API key: Set AUJ_TRADINGECONOMICS_API_KEY
# Rate limiting: Based on subscription
# Covers: Premium economic data with forecasts
# Data format: JSON API
```

---

## 🧪 Testing & Validation

### Comprehensive Test Suite

The economic calendar includes a comprehensive test suite covering all components:

```bash
# Run all economic calendar tests
cd tests
python test_economic_calendar_comprehensive.py

# Run specific test categories
python -m pytest test_economic_calendar_comprehensive.py::TestEconomicCalendarProviders -v
python -m pytest test_economic_calendar_comprehensive.py::TestEconomicIndicators -v
python -m pytest test_economic_calendar_comprehensive.py::TestEconomicDatabase -v
python -m pytest test_economic_calendar_comprehensive.py::TestEconomicMonitor -v
python -m pytest test_economic_calendar_comprehensive.py::TestEconomicIntegration -v
```

### Expected Test Results
```
Tests run: 45
Failures: 0
Errors: 0
Success rate: 100.0%
```

### Manual Testing

#### 1. Test Data Providers
```bash
python -c "
import sys
sys.path.append('auj_platform/src')

from analytics.providers.economic_calendar_providers import ForexFactoryProvider

# Test ForexFactory
ff = ForexFactoryProvider()
events = ff.get_economic_events()
print(f'ForexFactory: {len(events)} events fetched')

# Verify event structure
if events:
    event = events[0]
    required_fields = ['title', 'country', 'currency', 'impact_level', 'date_time']
    for field in required_fields:
        assert field in event, f'Missing field: {field}'
    print('✅ Event structure validated')
"
```

#### 2. Test Economic Indicators
```bash
python -c "
import sys
sys.path.append('auj_platform/src')
from datetime import datetime

from analytics.indicators.economic.economic_event_impact_indicator import EconomicEventImpactIndicator

# Test indicator
indicator = EconomicEventImpactIndicator()
sample_event = {
    'title': 'Non-Farm Payrolls',
    'currency': 'USD',
    'impact_level': 'High',
    'date_time': datetime.now().isoformat()
}

result = indicator.calculate({
    'economic_events': [sample_event],
    'current_time': datetime.now(),
    'currency_pair': 'EURUSD'
})

print(f'Indicator result: {result}')
assert 'signal_strength' in result
print('✅ Economic indicator validated')
"
```

#### 3. Test Database Operations
```bash
python -c "
import sys
sys.path.append('auj_platform/src')
from datetime import datetime
from database import DatabaseManager

# Test database
db = DatabaseManager()
sample_event = {
    'title': 'Test Event',
    'country': 'United States',
    'currency': 'USD',
    'date_time': datetime.now().isoformat(),
    'impact_level': 'High'
}

# Save event
event_id = db.save_economic_event(sample_event)
print(f'Event saved with ID: {event_id}')

# Retrieve events
events = db.get_economic_events()
print(f'Retrieved {len(events)} events')
print('✅ Database operations validated')
"
```

#### 4. Test Dashboard Access
```bash
# Start dashboard
cd auj_platform/dashboard
streamlit run app.py &

# Wait for startup
sleep 5

# Test dashboard access
curl -f http://localhost:8501 > /dev/null && echo "✅ Dashboard accessible" || echo "❌ Dashboard not accessible"

# Stop dashboard
pkill -f streamlit
```

### Performance Testing

```bash
# Run performance benchmarks
python tests/test_economic_calendar_comprehensive.py TestEconomicPerformance

# Expected results:
# - Bulk event insertion: < 10 seconds for 1000 events
# - Indicator calculations: < 5 seconds for 100 events
# - Database queries: < 1 second for complex queries
```

---

## 🚀 Production Deployment

### Docker Deployment (Recommended)

#### 1. Docker Compose Setup
```bash
# Ensure Docker and Docker Compose are installed
docker --version
docker-compose --version

# Start production services
docker-compose -f docker-compose.prod.yml up -d

# Verify services
docker-compose ps
```

#### Expected Services:
- `auj-platform`: Main application
- `postgres`: Database
- `rabbitmq`: Message broker
- `redis`: Cache
- `prometheus`: Metrics
- `grafana`: Monitoring dashboards

#### 2. Environment Configuration
```bash
# Create production environment file
cat > .env.prod << EOF
# Production Database
AUJ_DB_HOST=postgres
AUJ_DB_PORT=5432
AUJ_DB_NAME=auj_platform
AUJ_DB_USER=auj_user
AUJ_DB_PASS=$(openssl rand -base64 32)

# RabbitMQ
AUJ_RABBITMQ_HOST=rabbitmq
AUJ_RABBITMQ_USER=auj_user
AUJ_RABBITMQ_PASS=$(openssl rand -base64 32)

# Redis
AUJ_REDIS_HOST=redis
AUJ_REDIS_PASS=$(openssl rand -base64 32)

# Security
AUJ_ENCRYPTION_KEY=$(openssl rand -base64 32)
AUJ_JWT_SECRET=$(openssl rand -base64 32)

# Economic Calendar API Keys
AUJ_TRADINGECONOMICS_API_KEY=your_production_api_key
EOF

# Load production environment
docker-compose --env-file .env.prod -f docker-compose.prod.yml up -d
```

#### 3. SSL/TLS Setup
```bash
# Generate SSL certificates (or use Let's Encrypt)
sudo apt install certbot

# For domain-based setup
sudo certbot certonly --standalone -d your-domain.com

# Update nginx configuration
sudo vim /etc/nginx/sites-available/auj-platform

# Example nginx config:
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# Enable site and restart nginx
sudo ln -s /etc/nginx/sites-available/auj-platform /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```

### Manual Deployment

#### 1. System Service Setup
```bash
# Create systemd service for economic monitor
sudo tee /etc/systemd/system/auj-economic-monitor.service > /dev/null << EOF
[Unit]
Description=AUJ Economic Calendar Monitor
After=network.target

[Service]
Type=simple
User=auj
WorkingDirectory=/opt/auj-platform
ExecStart=/opt/auj-platform/venv/bin/python auj_platform/src/monitoring/economic_monitor.py --start
Restart=always
RestartSec=10
Environment=PYTHONPATH=/opt/auj-platform/auj_platform/src

[Install]
WantedBy=multi-user.target
EOF

# Create systemd service for dashboard
sudo tee /etc/systemd/system/auj-dashboard.service > /dev/null << EOF
[Unit]
Description=AUJ Platform Dashboard
After=network.target

[Service]
Type=simple
User=auj
WorkingDirectory=/opt/auj-platform/auj_platform/dashboard
ExecStart=/opt/auj-platform/venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10
Environment=PYTHONPATH=/opt/auj-platform/auj_platform/src

[Install]
WantedBy=multi-user.target
EOF

# Enable and start services
sudo systemctl daemon-reload
sudo systemctl enable auj-economic-monitor
sudo systemctl enable auj-dashboard
sudo systemctl start auj-economic-monitor
sudo systemctl start auj-dashboard

# Check status
sudo systemctl status auj-economic-monitor
sudo systemctl status auj-dashboard
```

#### 2. Database Migration
```bash
# Run production database migration
cd /opt/auj-platform/auj_platform
python src/database/migrate_economic_calendar_db.py --production

# Verify tables
python -c "
from src.database import DatabaseManager
db = DatabaseManager()
tables = db.get_table_names()
print('Production tables:', tables)
assert 'economic_events' in tables
print('✅ Production database ready')
"
```

#### 3. Load Balancer Setup (Optional)
```bash
# Install HAProxy for load balancing
sudo apt install haproxy

# Configure HAProxy
sudo tee /etc/haproxy/haproxy.cfg > /dev/null << EOF
frontend auj_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/auj-platform.pem
    redirect scheme https if !{ ssl_fc }
    default_backend auj_dashboard

backend auj_dashboard
    balance roundrobin
    server dashboard1 127.0.0.1:8501 check
    
backend auj_api
    balance roundrobin
    server api1 127.0.0.1:8000 check
EOF

# Start HAProxy
sudo systemctl restart haproxy
sudo systemctl enable haproxy
```

### Production Checklist

- [ ] ✅ SSL/TLS certificates configured
- [ ] ✅ Database connection secured
- [ ] ✅ Environment variables set
- [ ] ✅ Firewall configured (ports 80, 443 open)
- [ ] ✅ Backup strategy implemented
- [ ] ✅ Monitoring dashboards accessible
- [ ] ✅ Log rotation configured
- [ ] ✅ Economic calendar data flowing
- [ ] ✅ Real-time monitoring active
- [ ] ✅ Health checks passing

---

## 📊 Monitoring & Maintenance

### Health Monitoring

#### 1. Application Health Checks
```bash
# Economic calendar health check
curl -f http://localhost:8000/api/v1/economic/health || echo "Economic calendar not responding"

# Database health check
python -c "
from auj_platform.src.database import DatabaseManager
db = DatabaseManager()
try:
    events = db.get_economic_events(limit=1)
    print('✅ Database healthy')
except Exception as e:
    print(f'❌ Database error: {e}')
"

# Dashboard health check
curl -f http://localhost:8501/_stcore/health || echo "Dashboard not responding"
```

#### 2. Prometheus Metrics

Key metrics to monitor:

```yaml
# Economic calendar specific metrics
economic_events_fetched_total: Counter of events fetched from providers
economic_signals_generated_total: Counter of trading signals generated
economic_indicator_calculation_duration: Histogram of indicator calculation times
economic_provider_errors_total: Counter of provider errors
economic_database_operations_total: Counter of database operations

# System metrics
memory_usage_bytes: Current memory usage
cpu_usage_percent: Current CPU usage
database_connections_active: Active database connections
message_queue_length: RabbitMQ queue lengths
```

#### 3. Grafana Dashboards

Access Grafana at `http://your-domain:3000` with these dashboards:

- **Economic Calendar Overview**: Events, signals, provider status
- **Performance Metrics**: Indicator calculation times, database performance
- **Alert Dashboard**: Real-time alerts and notifications
- **System Health**: CPU, memory, disk, network usage

### Log Management

#### 1. Log Locations
```bash
# Application logs
tail -f auj_platform/logs/auj_platform.log

# Economic monitor logs
tail -f auj_platform/logs/economic_monitor.log

# Dashboard logs
tail -f auj_platform/logs/dashboard.log

# Error logs
tail -f auj_platform/logs/auj_platform_errors.log
```

#### 2. Log Rotation Setup
```bash
# Create logrotate configuration
sudo tee /etc/logrotate.d/auj-platform > /dev/null << EOF
/opt/auj-platform/auj_platform/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 auj auj
    postrotate
        systemctl reload auj-economic-monitor
        systemctl reload auj-dashboard
    endscript
}
EOF

# Test logrotate
sudo logrotate -d /etc/logrotate.d/auj-platform
```

### Backup Strategy

#### 1. Database Backup
```bash
# Create backup script
cat > /opt/auj-platform/scripts/backup_database.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/auj-platform/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/auj_platform_$DATE.sql"

mkdir -p $BACKUP_DIR

# PostgreSQL backup
pg_dump -h localhost -U auj_user auj_platform > $BACKUP_FILE

# Compress backup
gzip $BACKUP_FILE

# Remove backups older than 30 days
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_FILE.gz"
EOF

chmod +x /opt/auj-platform/scripts/backup_database.sh

# Schedule daily backups
echo "0 2 * * * /opt/auj-platform/scripts/backup_database.sh" | sudo crontab -u auj -
```

#### 2. Configuration Backup
```bash
# Create configuration backup script
cat > /opt/auj-platform/scripts/backup_config.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/auj-platform/backups/config"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup configuration files
tar -czf "$BACKUP_DIR/config_$DATE.tar.gz" \
    /opt/auj-platform/config/ \
    /opt/auj-platform/.env* \
    /etc/systemd/system/auj-* \
    /etc/nginx/sites-available/auj-platform

echo "Configuration backup completed: config_$DATE.tar.gz"
EOF

chmod +x /opt/auj-platform/scripts/backup_config.sh
```

### Maintenance Tasks

#### Daily Maintenance
```bash
# Check system status
sudo systemctl status auj-economic-monitor auj-dashboard

# Check disk space
df -h /opt/auj-platform

# Check recent errors
grep -i error auj_platform/logs/auj_platform.log | tail -10

# Verify economic data freshness
python -c "
from auj_platform.src.database import DatabaseManager
from datetime import datetime, timedelta
db = DatabaseManager()
recent_events = db.get_economic_events(
    start_date=datetime.now() - timedelta(hours=24)
)
print(f'Events in last 24h: {len(recent_events)}')
"
```

#### Weekly Maintenance
```bash
# Update economic calendar dependencies
pip install --upgrade beautifulsoup4 requests lxml

# Analyze performance trends
python auj_platform/src/analytics/performance_analyzer.py --weekly-report

# Clean old logs
find auj_platform/logs -name "*.log.*" -mtime +7 -delete

# Optimize database
python -c "
from auj_platform.src.database import DatabaseManager
db = DatabaseManager()
db.optimize_tables()
print('Database optimization completed')
"
```

#### Monthly Maintenance
```bash
# Full system backup
/opt/auj-platform/scripts/backup_database.sh
/opt/auj-platform/scripts/backup_config.sh

# Security updates
sudo apt update && sudo apt upgrade -y

# Certificate renewal (if using Let's Encrypt)
sudo certbot renew

# Performance review
python auj_platform/src/analytics/performance_analyzer.py --monthly-report
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Economic Data Not Loading

**Symptoms**: Dashboard shows "No economic events found"

**Diagnosis**:
```bash
# Check provider connectivity
python -c "
import sys
sys.path.append('auj_platform/src')
from analytics.providers.economic_calendar_providers import ForexFactoryProvider

ff = ForexFactoryProvider()
try:
    events = ff.get_economic_events()
    print(f'✅ ForexFactory: {len(events)} events')
except Exception as e:
    print(f'❌ ForexFactory error: {e}')
"
```

**Solutions**:
- Check internet connectivity
- Verify provider websites are accessible
- Check rate limiting (wait 60 seconds between requests)
- Update user agent string in configuration

#### 2. Database Connection Issues

**Symptoms**: "Database connection failed" errors

**Diagnosis**:
```bash
# Test database connection
python -c "
from auj_platform.src.database import DatabaseManager
try:
    db = DatabaseManager()
    db.get_connection()
    print('✅ Database connection successful')
except Exception as e:
    print(f'❌ Database error: {e}')
"
```

**Solutions**:
- Verify database service is running: `sudo systemctl status postgresql`
- Check connection string in configuration
- Verify credentials and permissions
- Check firewall settings

#### 3. Dashboard Loading Issues

**Symptoms**: Dashboard not accessible or loading slowly

**Diagnosis**:
```bash
# Check dashboard process
ps aux | grep streamlit

# Check port availability
netstat -tulpn | grep :8501

# Check dashboard logs
tail -f auj_platform/logs/dashboard.log
```

**Solutions**:
- Restart dashboard service: `sudo systemctl restart auj-dashboard`
- Check memory usage: `free -h`
- Clear browser cache
- Verify firewall allows port 8501

#### 4. Economic Indicators Not Calculating

**Symptoms**: Indicator results show zero signal strength

**Diagnosis**:
```bash
# Test individual indicators
python -c "
import sys
sys.path.append('auj_platform/src')
from analytics.indicators.economic.economic_event_impact_indicator import EconomicEventImpactIndicator
from datetime import datetime

indicator = EconomicEventImpactIndicator()
sample_data = {
    'economic_events': [{'title': 'Test', 'impact_level': 'High', 'currency': 'USD'}],
    'current_time': datetime.now(),
    'currency_pair': 'EURUSD'
}

try:
    result = indicator.calculate(sample_data)
    print(f'✅ Indicator result: {result}')
except Exception as e:
    print(f'❌ Indicator error: {e}')
"
```

**Solutions**:
- Verify economic events are properly formatted
- Check date/time parsing
- Ensure required fields are present
- Update indicator calculation logic

#### 5. Real-time Monitoring Not Working

**Symptoms**: No real-time alerts or signals

**Diagnosis**:
```bash
# Check monitoring service
sudo systemctl status auj-economic-monitor

# Check monitoring logs
tail -f auj_platform/logs/economic_monitor.log

# Test monitoring manually
python auj_platform/src/monitoring/economic_monitor.py --test
```

**Solutions**:
- Restart monitoring service: `sudo systemctl restart auj-economic-monitor`
- Check message broker connection
- Verify alert thresholds in configuration
- Check system resources

### Performance Issues

#### High Memory Usage
```bash
# Monitor memory usage
htop

# Check Python process memory
ps aux --sort=-%mem | head

# Optimize database queries
python -c "
from auj_platform.src.database import DatabaseManager
db = DatabaseManager()
db.analyze_query_performance()
"
```

#### Slow Dashboard Response
```bash
# Check dashboard performance
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8501"

# Optimize dashboard caching
# Edit auj_platform/dashboard/app.py
# Add @st.cache_data decorators to expensive functions
```

#### Database Performance Issues
```bash
# Check database statistics
python -c "
from auj_platform.src.database import DatabaseManager
db = DatabaseManager()
stats = db.get_database_statistics()
print('Database stats:', stats)
"

# Add database indexes
python -c "
from auj_platform.src.database import DatabaseManager
db = DatabaseManager()
db.create_performance_indexes()
print('Performance indexes created')
"
```

### Getting Help

#### Log Collection for Support
```bash
# Create support package
mkdir -p /tmp/auj-support
cp -r auj_platform/logs /tmp/auj-support/
cp config/main_config.yaml /tmp/auj-support/
cp .env /tmp/auj-support/ 2>/dev/null || echo "No .env file"

# System information
uname -a > /tmp/auj-support/system_info.txt
python --version >> /tmp/auj-support/system_info.txt
pip list > /tmp/auj-support/pip_packages.txt

# Create archive
tar -czf auj-support-$(date +%Y%m%d).tar.gz -C /tmp auj-support
echo "Support package created: auj-support-$(date +%Y%m%d).tar.gz"
```

#### Community Resources
- **Documentation**: `/docs` directory
- **GitHub Issues**: Report bugs and feature requests
- **Discord**: Real-time community support
- **Email Support**: technical-support@auj-platform.com

---

## 📋 API Reference

### Economic Calendar Endpoints

#### Get Economic Events
```http
GET /api/v1/economic/events

Query Parameters:
- start_date (string): ISO format date
- end_date (string): ISO format date  
- impact_level (string): High, Medium, Low
- currency (string): USD, EUR, GBP, etc.
- limit (integer): Max events to return

Response:
{
  "events": [
    {
      "id": "string",
      "title": "Non-Farm Payrolls",
      "country": "United States",
      "currency": "USD",
      "date_time": "2024-01-15T08:30:00Z",
      "impact_level": "High",
      "actual_value": "150K",
      "forecast_value": "145K",
      "previous_value": "140K"
    }
  ],
  "total": 150,
  "page": 1
}
```

#### Get Trading Signals
```http
GET /api/v1/economic/signals

Query Parameters:
- currency_pair (string): EURUSD, GBPUSD, etc.
- signal_type (string): BUY, SELL
- limit (integer): Max signals to return

Response:
{
  "signals": [
    {
      "id": "string",
      "currency_pair": "EURUSD",
      "signal_type": "BUY",
      "signal_strength": 0.85,
      "confidence_score": 0.78,
      "timestamp": "2024-01-15T09:00:00Z",
      "related_events": ["event_id_1", "event_id_2"]
    }
  ]
}
```

#### Get Performance Data
```http
GET /api/v1/economic/performance

Response:
{
  "overall_performance": {
    "total_signals": 150,
    "accuracy_rate": 0.73,
    "total_pnl": 1250.50,
    "avg_signal_strength": 0.67
  },
  "by_indicator": {
    "event_impact": {
      "accuracy_rate": 0.75,
      "signal_count": 45
    }
  }
}
```

### Health Check Endpoints

#### System Health
```http
GET /api/v1/health

Response:
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:00:00Z",
  "components": {
    "database": "healthy",
    "economic_providers": "healthy",
    "indicators": "healthy",
    "monitoring": "healthy"
  }
}
```

#### Economic Calendar Health
```http
GET /api/v1/economic/health

Response:
{
  "status": "healthy",
  "last_update": "2024-01-15T09:45:00Z",
  "providers": {
    "forexfactory": "healthy",
    "investing": "healthy",
    "tradingeconomics": "api_key_required"
  },
  "indicators": {
    "event_impact": "healthy",
    "volatility_predictor": "healthy",
    "confluence": "healthy"
  }
}
```

---

## 🎉 Conclusion

Congratulations! You now have a fully functional, production-ready economic calendar system integrated with the AUJ Platform. The system provides:

✅ **Real-time economic event monitoring**  
✅ **Automated trading signal generation**  
✅ **Comprehensive analytics and performance tracking**  
✅ **Production-grade monitoring and alerting**  
✅ **Scalable architecture with Docker support**  

### Next Steps

1. **🔍 Monitor Performance**: Use Grafana dashboards to track system performance
2. **📈 Optimize Strategies**: Analyze economic signal performance and refine indicators
3. **🔧 Customize Configuration**: Adjust thresholds and parameters based on your trading style
4. **📱 Set Up Alerts**: Configure notifications for high-impact economic events
5. **🚀 Scale Infrastructure**: Add more providers and expand to additional markets

### Support & Resources

- **📖 Documentation**: Complete documentation in `/docs` directory
- **🧪 Testing**: Comprehensive test suite in `/tests` directory  
- **📊 Monitoring**: Grafana dashboards for system monitoring
- **🔧 Configuration**: Flexible YAML-based configuration system
- **🐛 Troubleshooting**: Detailed troubleshooting guide above

**Happy Trading!** 🚀📈

---

*Last Updated: January 2024*  
*Version: 2.0.0*  
*Platform: AUJ Economic Calendar*