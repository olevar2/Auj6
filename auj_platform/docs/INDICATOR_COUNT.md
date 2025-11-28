# 📊 Indicator Count Reference

**Last Updated:** 2025-11-28  
**Status:** ✅ Verified with physical files

---

## Official Count

### Total Indicators: **159**

This is the **official, verified count** of indicators in the AUJ Platform.

---

## Breakdown by Agent

| Agent | Indicator Count |
|-------|----------------|
| TrendAgent | 29 |
| DecisionMaster | 23 |
| ExecutionExpert | 18 |
| RiskGenius | 17 |
| MicrostructureAgent | 16 |
| IndicatorExpert | 15 |
| PatternMaster | 12 |
| SessionExpert | 12 |
| PairSpecialist | 11 |
| MomentumAgent | 8 |
| **TOTAL** | **159** |

---

## Verification Details

### Physical Files
- **Implementation Files:** 164 `.py` files
- **Assigned Indicators:** 159 unique indicators
- **Status:** ✅ All 159 have corresponding implementation files

### Source of Truth
The **single source of truth** for indicator assignments is:
```
src/registry/agent_indicator_mapping.py
```

This file contains the `AGENT_MAPPINGS` dictionary with all 159 indicators distributed across 10 expert agents.

---

## Important Rules

### ⚠️ DO NOT Change This Number Without:

1. **Updating the Mapping File**
   - Modify `src/registry/agent_indicator_mapping.py`
   - Update the documentation block at the top

2. **Creating/Removing Physical Files**
   - Add/remove indicator implementation in `src/indicator_engine/indicators/`
   - Ensure IndicatorRegistry can discover the new files

3. **Running Validation**
   ```python
   from registry.agent_indicator_mapping import validate_agent_mapping
   result = validate_agent_mapping()
   print(result)
   # Should show: unique_indicators = 159
   ```

4. **Updating This Document**
   - Update the count in this file
   - Update the "Last Updated" date
   - Document what was added/removed

---

## Verification Commands

### Get Current Count
```python
from registry.agent_indicator_mapping import get_all_mapped_indicators
indicators = get_all_mapped_indicators()
print(f"Total: {len(indicators)}")
# Expected output: Total: 159
```

### Validate Mapping
```python
from registry.agent_indicator_mapping import validate_agent_mapping
summary = validate_agent_mapping()
print(summary)
# Expected output:
# {
#   'unique_indicators': 159,
#   'total_assignments': 159,
#   'agents_count': 10,
#   'mapping_complete': True
# }
```

### Get Summary
```python
from registry.agent_indicator_mapping import get_mapping_summary
summary = get_mapping_summary()
print(summary['TOTAL'])
# Expected output:
# {
#   'unique_indicators': 159,
#   'total_assignments': 159,
#   'agents': 10
# }
```

---

## History

### 2025-11-28
- **Verified Count:** 159 indicators
- **Physical Files:** 164 files (includes 5 utility/base files)
- **Status:** All 159 assigned indicators have implementation files
- **Documentation:** Added this reference file

---

## Directory Structure

```
src/indicator_engine/indicators/
├── ai_enhanced/       (32 files)
├── trend/            (29 files)
├── volume/           (29 files)
├── statistical/      (17 files)
├── momentum/         (12 files)
├── gann/             (12 files)
├── other/            (9 files)
├── pattern/          (8 files)
├── volatility/       (8 files)
├── fibonacci/        (6 files)
├── elliott_wave/     (2 files)
└── base/             (2 files - utilities)
```

**Total Physical Files:** 164 (includes base/utility files)  
**Total Assigned Indicators:** 159

---

## Reference Files

- **Mapping:** `src/registry/agent_indicator_mapping.py`
- **Registry:** `src/registry/indicator_registry.py`
- **Executor:** `src/indicator_engine/indicator_executor.py`
- **This Document:** `docs/INDICATOR_COUNT.md`

---

**Maintained by:** Platform Development Team  
**Contact:** For questions about indicator count, verify with `agent_indicator_mapping.py`
