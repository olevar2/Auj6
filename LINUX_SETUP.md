# AUJ Platform - Linux MetaApi Setup

Quick setup guide for running AUJ Platform on Linux with MetaApi integration.

## Quick Start

### 1. Prerequisites

- Linux (Ubuntu 20.04+ recommended)
- Python 3.9+
- MetaApi account and token

### 2. Installation

```bash
# Clone and setup
git clone <repository-url>
cd AUJ

# Run quick setup
chmod +x setup_linux_quick.sh
./setup_linux_quick.sh
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit with your credentials
nano .env
```

Required variables:

```bash
AUJ_METAAPI_TOKEN=your_metaapi_token_here
AUJ_METAAPI_ACCOUNT_ID=your_metaapi_account_id_here
```

### 4. Test Installation

```bash
# Activate virtual environment
source venv/bin/activate

# Test MetaApi integration
python3 test_metaapi.py

# Test platform launcher
python3 run_linux.py
```

### 5. Run Platform

```bash
# Dashboard
python3 -m auj_platform.dashboard.app

# API (in another terminal)
python3 -m auj_platform.src.api.main
```

## Key Files Created

- `run_linux.py` - Platform launcher and validator
- `test_metaapi.py` - MetaApi integration tester
- `setup_linux_quick.sh` - Quick installation script
- `.env.example` - Environment variables template
- `requirements-linux.txt` - Linux-specific dependencies

## MetaApi Integration

The platform automatically:

- Detects Linux environment
- Prioritizes MetaApi over MT5 direct
- Configures cross-platform data providers
- Enables WebSocket streaming for real-time data

## Troubleshooting

### Common Issues:

1. **Missing MetaApi credentials:**

   ```bash
   export AUJ_METAAPI_TOKEN=your_token
   export AUJ_METAAPI_ACCOUNT_ID=your_account_id
   ```

2. **Import errors:**

   ```bash
   pip install -r requirements-linux.txt
   ```

3. **Permission errors:**
   ```bash
   chmod +x setup_linux_quick.sh
   chmod +x run_linux.py
   ```

## Next Steps

1. Configure trading parameters in `config/metaapi_config.yaml`
2. Set up monitoring (optional)
3. Deploy to production server

For detailed configuration, see `config/README.md`.
