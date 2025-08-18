#!/usr/bin/env python3
"""Validation script for agent indicator mapping."""

from agent_indicator_mapping import get_mapping_summary

def main():
    summary = get_mapping_summary()
    print("FINAL MAPPING STATUS:")
    print("=" * 50)

    for agent, data in summary.items():
        if agent != "TOTAL":
            print(f"{agent}: {data['indicator_count']} indicators")
        else:
            print("=" * 50)
            print(f"TOTAL UNIQUE INDICATORS: {data['unique_indicators']}")
            print(f"TOTAL ASSIGNMENTS: {data['total_assignments']}")
            print(f"AGENTS: {data['agents']}")

            if data['total_assignments'] > data['unique_indicators']:
                duplicates = data['total_assignments'] - data['unique_indicators']
                print(f"REMAINING DUPLICATES: {duplicates}")

if __name__ == "__main__":
    main()
