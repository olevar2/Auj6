import yaml
from pathlib import Path

config_dir = Path('config')

# Test metaapi_config_NEW.yaml
print("Testing MetaApi Configuration...")
with open(config_dir / 'metaapi_config_NEW.yaml', 'r', encoding='utf-8') as f:
    metaapi_data = yaml.safe_load(f)
print("Valid YAML")
print(f"MetaApi enabled: {metaapi_data.get('metaapi', {}).get('enabled')}")

print()

# Test linux_deployment_NEW.yaml
print("Testing Linux Deployment Configuration...")
with open(config_dir / 'linux_deployment_NEW.yaml', 'r', encoding='utf-8') as f:
    linux_data = yaml.safe_load(f)
print("Valid YAML")
print(f"Platform: {linux_data.get('deployment', {}).get('platform')}")
print(f"Primary provider: {linux_data.get('data_providers', {}).get('primary')}")

print()
print("All configuration files validated successfully!")
