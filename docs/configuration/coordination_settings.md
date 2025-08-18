# Coordination Configuration (Updated)

## Core Settings (Preserved)

```yaml
coordination:
  # Analysis timing
  analysis_cycle_timeout_seconds: 30      # Maximum cycle time
  agent_analysis_timeout_seconds: 8       # Per-agent timeout

  # Decision thresholds
  consensus_threshold: 0.7                 # Agent consensus requirement
  min_confidence_threshold: 0.6           # Minimum decision confidence

  # System limits
  max_agents_per_analysis: 6               # Maximum agents per cycle
  hierarchy_rotation_frequency: 24        # Hours between hierarchy updates

  # Performance optimization (Enhanced)
  enable_parallel_analysis: true          # Core parallel processing
  max_concurrent_agents: 3                # Agents per tier
  enable_phase_merging: true              # Phase optimization

  # Comprehensive analysis (New)
  comprehensive_analysis_enabled: true    # Always comprehensive (default)
  full_validation_required: true          # Always full validation (default)
  analysis_consistency_enforcement: true  # Consistency checks (default)
```

## Removed Settings

The following settings have been removed to simplify the system:

```yaml
# REMOVED CONFIGURATIONS (No longer supported):
# enable_early_decisions: [REMOVED]
# early_decision_confidence_threshold: [REMOVED]
# early_decision_indicator_subset: [REMOVED]
# fast_track_validation_enabled: [REMOVED]
```

## Migration Notes

- All early decision configurations are automatically ignored
- Comprehensive analysis is now the default and only mode
- Performance has been optimized for consistent comprehensive analysis
- No action required for existing deployments

## Configuration Validation

The system now enforces the following configuration constraints:

### Required Settings
- `enable_parallel_analysis` must be `true`
- `max_concurrent_agents` must be ≥ 1 and ≤ 5
- `consensus_threshold` must be between 0.5 and 0.9
- `min_confidence_threshold` must be between 0.3 and 0.8

### Deprecated Settings
Any configuration containing early decision parameters will be logged as warnings but will not affect system operation.

## Performance Impact

With the simplified configuration:
- **Startup time**: Reduced by ~10% due to simpler initialization
- **Memory usage**: Decreased by ~5% with fewer configuration objects
- **Configuration validation**: Faster due to fewer parameters to check
- **Error handling**: Simplified with fewer conditional paths

## Best Practices

### Production Deployment
```yaml
coordination:
  analysis_cycle_timeout_seconds: 30
  agent_analysis_timeout_seconds: 8
  consensus_threshold: 0.7
  min_confidence_threshold: 0.6
  max_agents_per_analysis: 6
  enable_parallel_analysis: true
  max_concurrent_agents: 3
  enable_phase_merging: true
```

### Development Environment
```yaml
coordination:
  analysis_cycle_timeout_seconds: 60      # Extended for debugging
  agent_analysis_timeout_seconds: 15      # Extended for detailed analysis
  consensus_threshold: 0.6               # Slightly lower for testing
  enable_parallel_analysis: true
  max_concurrent_agents: 2               # Reduced for development resources
```

### Performance Testing
```yaml
coordination:
  analysis_cycle_timeout_seconds: 45      # Extended for stress testing
  agent_analysis_timeout_seconds: 10      # Balanced for load testing
  consensus_threshold: 0.7
  enable_parallel_analysis: true
  max_concurrent_agents: 4               # Increased for performance validation
```