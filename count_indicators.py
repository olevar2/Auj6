"""
Script to count actual indicators used in the platform
"""

from auj_platform.src.registry.agent_indicator_mapping import AGENT_MAPPINGS, get_all_mapped_indicators

# Count indicators assigned to each agent
print("=" * 60)
print("Indicators per Agent:")
print("=" * 60)

total_assignments = 0
for agent_name, agent_data in AGENT_MAPPINGS.items():
    count = len(agent_data["assigned_indicators"])
    total_assignments += count
    print(f"{agent_name:25} : {count:3} indicators")

print("=" * 60)

# Get unique indicators
unique_indicators = get_all_mapped_indicators()
print(f"\nTotal assignments (with duplicates): {total_assignments}")
print(f"Unique indicators actually used: {len(unique_indicators)}")
print(f"Number of agents: {len(AGENT_MAPPINGS)}")

print("\n" + "=" * 60)
print("FINAL STATS:")
print("=" * 60)
print(f"ACTUAL INDICATORS IN USE: {len(unique_indicators)} indicators")
print(f"Total assignments: {total_assignments}")
print(f"Number of agents: {len(AGENT_MAPPINGS)}")
print(f"Average indicators per agent: {total_assignments / len(AGENT_MAPPINGS):.1f}")

# Print first 10 indicators as sample
print("\n" + "=" * 60)
print("Sample indicators (first 10):")
print("=" * 60)
for i, indicator in enumerate(unique_indicators[:10], 1):
    print(f"{i:2}. {indicator}")

print(f"\n... and {len(unique_indicators) - 10} more indicators")
