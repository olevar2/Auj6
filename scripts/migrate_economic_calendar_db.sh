#!/bin/bash
echo "=========================================="
echo "AUJ Platform Economic Calendar Migration"
echo "=========================================="
echo

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Running database migration for economic calendar..."
python3 scripts/migrate_economic_calendar_db.py

echo
echo "Migration script completed."
echo "Check the logs above for results."
echo