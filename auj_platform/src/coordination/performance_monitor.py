"""
Performance monitoring system for AUJ Platform coordination.

This module provides comprehensive performance tracking for the coordination system,
monitoring cycle times, phase performance, parallel efficiency, and optimization metrics.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import statistics
import logging

logger = logging.getLogger(__name__)


@dataclass
class PhasePerformance:
    """Performance metrics for a single phase."""
    phase_name: str
    duration_seconds: float
    start_time: datetime
    end_time: datetime
    success: bool
    parallel_operations: int = 0
    early_exit: bool = False
    error_message: Optional[str] = None


@dataclass
class CyclePerformance:
    """Performance metrics for a complete analysis cycle."""
    cycle_id: str
    total_duration_seconds: float
    start_time: datetime
    end_time: datetime
    success: bool
    agents_processed: int
    indicators_calculated: int
    phases: List[PhasePerformance] = field(default_factory=list)

    # COMPREHENSIVE ANALYSIS tracking:
    comprehensive_analysis_applied: bool = True
    full_validation_completed: bool = False
    analysis_consistency_score: float = 1.0

    # PRESERVE and ENHANCE:
    parallel_efficiency: float = 0.0
    signal_generated: bool = False
    optimization_flags: Dict[str, bool] = field(default_factory=dict)


class PerformanceMetrics(Enum):
    """Performance metric types."""
    CYCLE_TIME = "cycle_time"
    PHASE_TIME = "phase_time"
    PARALLEL_EFFICIENCY = "parallel_efficiency"
    SUCCESS_RATE = "success_rate"
    THROUGHPUT = "throughput"

    # ADD:
    COMPREHENSIVE_ANALYSIS_RATE = "comprehensive_analysis_rate"
    ANALYSIS_CONSISTENCY = "analysis_consistency"


class CoordinationPerformanceMonitor:
    """Monitor performance improvements in coordination."""

    def __init__(self, max_history: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            max_history: Maximum number of cycles to keep in history
        """
        self.max_history = max_history
        
        # Performance data storage
        self.cycle_history: List[CyclePerformance] = []
        self.phase_times: Dict[str, List[float]] = {}
        self.parallel_efficiency_history: List[float] = []
        
        # Current cycle tracking
        self.current_cycle: Optional[CyclePerformance] = None
        self.current_phase: Optional[PhasePerformance] = None
        
        # Performance statistics
        self.baseline_cycle_time = 1800.0  # 30 minutes baseline
        self.target_cycle_time = 300.0     # 5 minutes target
        
        # Monitoring start time
        self.monitoring_start_time = datetime.utcnow()

    def start_cycle_monitoring(self, cycle_id: str, optimization_flags: Dict[str, bool] = None) -> None:
        """Start monitoring a new analysis cycle."""
        if self.current_cycle is not None:
            logger.warning(f"Starting new cycle {cycle_id} while cycle {self.current_cycle.cycle_id} is still active")
            self.end_cycle_monitoring(success=False, error="Interrupted by new cycle")

        self.current_cycle = CyclePerformance(
            cycle_id=cycle_id,
            total_duration_seconds=0.0,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            success=False,
            agents_processed=0,
            indicators_calculated=0,
            optimization_flags=optimization_flags or {}
        )
        
        logger.debug(f"Started monitoring cycle {cycle_id}")

    def start_phase_monitoring(self, phase_name: str, parallel_operations: int = 0) -> None:
        """Start monitoring a phase within the current cycle."""
        if self.current_cycle is None:
            logger.warning(f"Cannot start phase {phase_name} monitoring without active cycle")
            return

        if self.current_phase is not None:
            logger.warning(f"Starting phase {phase_name} while phase {self.current_phase.phase_name} is still active")
            self.end_phase_monitoring(success=False, error="Interrupted by new phase")

        self.current_phase = PhasePerformance(
            phase_name=phase_name,
            duration_seconds=0.0,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            success=False,
            parallel_operations=parallel_operations
        )
        
        logger.debug(f"Started monitoring phase {phase_name} with {parallel_operations} parallel operations")

    def end_phase_monitoring(self, success: bool = True, early_exit: bool = False, 
                           error: str = None) -> None:
        """End monitoring of the current phase."""
        if self.current_phase is None:
            logger.warning("Cannot end phase monitoring - no active phase")
            return

        end_time = datetime.utcnow()
        self.current_phase.end_time = end_time
        self.current_phase.duration_seconds = (end_time - self.current_phase.start_time).total_seconds()
        self.current_phase.success = success
        self.current_phase.early_exit = early_exit
        self.current_phase.error_message = error

        # Add to current cycle
        if self.current_cycle:
            self.current_cycle.phases.append(self.current_phase)
            
            # Track comprehensive analysis (always true now)
            if not early_exit:  # comprehensive analysis completed
                self.current_cycle.comprehensive_analysis_applied = True
                self.current_cycle.analysis_consistency_score = 1.0

        # Update phase statistics
        phase_name = self.current_phase.phase_name
        if phase_name not in self.phase_times:
            self.phase_times[phase_name] = []
        self.phase_times[phase_name].append(self.current_phase.duration_seconds)

        logger.debug(f"Ended monitoring phase {phase_name} - "
                    f"Duration: {self.current_phase.duration_seconds:.2f}s, "
                    f"Success: {success}, Early exit: {early_exit}")

        self.current_phase = None

    def end_cycle_monitoring(self, success: bool = True, agents_processed: int = 0,
                           indicators_calculated: int = 0, signal_generated: bool = False,
                           error: str = None) -> None:
        """End monitoring of the current cycle."""
        if self.current_cycle is None:
            logger.warning("Cannot end cycle monitoring - no active cycle")
            return

        # End any active phase
        if self.current_phase is not None:
            self.end_phase_monitoring(success=success, error="Cycle ended")

        end_time = datetime.utcnow()
        self.current_cycle.end_time = end_time
        self.current_cycle.total_duration_seconds = (end_time - self.current_cycle.start_time).total_seconds()
        self.current_cycle.success = success
        self.current_cycle.agents_processed = agents_processed
        self.current_cycle.indicators_calculated = indicators_calculated
        self.current_cycle.signal_generated = signal_generated

        # Calculate parallel efficiency
        self.current_cycle.parallel_efficiency = self._calculate_parallel_efficiency(self.current_cycle)
        
        # Store in history
        self.cycle_history.append(self.current_cycle)
        
        # Update parallel efficiency history
        if self.current_cycle.parallel_efficiency > 0:
            self.parallel_efficiency_history.append(self.current_cycle.parallel_efficiency)

        # Maintain history size
        if len(self.cycle_history) > self.max_history:
            self.cycle_history.pop(0)
        if len(self.parallel_efficiency_history) > self.max_history:
            self.parallel_efficiency_history.pop(0)

        logger.info(f"Completed monitoring cycle {self.current_cycle.cycle_id} - "
                   f"Duration: {self.current_cycle.total_duration_seconds:.2f}s, "
                   f"Success: {success}, Agents: {agents_processed}, "
                   f"Indicators: {indicators_calculated}, Signal: {signal_generated}")

        self.current_cycle = None

    def record_cycle_performance(self, cycle_state) -> None:
        """Record performance metrics for a cycle state (compatibility method)."""
        if hasattr(cycle_state, 'duration_seconds'):
            duration = cycle_state.duration_seconds
        else:
            duration = 0.0

        # Extract performance data
        agents_processed = len(getattr(cycle_state, 'agent_decisions', {}))
        indicators_calculated = len(getattr(cycle_state, 'performance_data', {}).get('indicator_results', {}))
        signal_generated = getattr(cycle_state, 'final_signal', None) is not None
        success = getattr(cycle_state, 'is_successful', False)

        # Use end cycle monitoring if no active monitoring
        if self.current_cycle is None:
            self.start_cycle_monitoring(getattr(cycle_state, 'cycle_id', 'unknown'))

        self.end_cycle_monitoring(
            success=success,
            agents_processed=agents_processed,
            indicators_calculated=indicators_calculated,
            signal_generated=signal_generated
        )

    def _calculate_parallel_efficiency(self, cycle: CyclePerformance) -> float:
        """Calculate parallel processing efficiency."""
        try:
            # Find agent analysis phase
            agent_phase = None
            for phase in cycle.phases:
                if 'agent' in phase.phase_name.lower() and 'analysis' in phase.phase_name.lower():
                    agent_phase = phase
                    break

            if not agent_phase or cycle.agents_processed == 0:
                return 0.0

            # Calculate theoretical sequential time (2 minutes per agent)
            theoretical_sequential_time = cycle.agents_processed * 120.0
            actual_time = agent_phase.duration_seconds

            if actual_time <= 0:
                return 0.0

            # Efficiency = theoretical_time / actual_time
            efficiency = min(theoretical_sequential_time / actual_time, 10.0)  # Cap at 10x
            return efficiency

        except Exception as e:
            logger.warning(f"Failed to calculate parallel efficiency: {str(e)}")
            return 0.0

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        if not self.cycle_history:
            return {'status': 'no_data', 'message': 'No performance data available'}

        try:
            # Basic statistics
            recent_cycles = self.cycle_history[-50:]  # Last 50 cycles
            all_durations = [cycle.total_duration_seconds for cycle in self.cycle_history if cycle.success]
            recent_durations = [cycle.total_duration_seconds for cycle in recent_cycles if cycle.success]

            if not all_durations:
                return {'status': 'no_successful_cycles', 'message': 'No successful cycles to analyze'}

            # Performance metrics
            avg_cycle_time = statistics.mean(all_durations)
            recent_avg_cycle_time = statistics.mean(recent_durations) if recent_durations else avg_cycle_time
            improvement = ((self.baseline_cycle_time - avg_cycle_time) / self.baseline_cycle_time) * 100
            recent_improvement = ((self.baseline_cycle_time - recent_avg_cycle_time) / self.baseline_cycle_time) * 100

            # Success rate
            total_cycles = len(self.cycle_history)
            successful_cycles = len([c for c in self.cycle_history if c.success])
            success_rate = successful_cycles / total_cycles if total_cycles > 0 else 0

            # ADD comprehensive analysis metrics:
            comprehensive_cycles = len([c for c in self.cycle_history if c.comprehensive_analysis_applied])
            comprehensive_rate = comprehensive_cycles / total_cycles if total_cycles > 0 else 1.0
            consistency_avg = statistics.mean([c.analysis_consistency_score for c in self.cycle_history if c.success]) if successful_cycles else 1.0

            # Parallel efficiency
            avg_parallel_efficiency = (
                statistics.mean(self.parallel_efficiency_history) 
                if self.parallel_efficiency_history else 0
            )

            # Phase performance
            phase_performance = {}
            for phase_name, times in self.phase_times.items():
                if times:
                    phase_performance[phase_name] = {
                        'average_seconds': statistics.mean(times),
                        'min_seconds': min(times),
                        'max_seconds': max(times),
                        'samples': len(times)
                    }

            # Target achievement
            target_achievement = (
                len([d for d in recent_durations if d <= self.target_cycle_time]) / 
                len(recent_durations) if recent_durations else 0
            ) * 100

            # Throughput (cycles per hour)
            monitoring_duration_hours = (datetime.utcnow() - self.monitoring_start_time).total_seconds() / 3600
            throughput = total_cycles / monitoring_duration_hours if monitoring_duration_hours > 0 else 0

            return {
                'status': 'success',
                'timestamp': datetime.utcnow().isoformat(),
                'monitoring_duration_hours': monitoring_duration_hours,
                
                # Core metrics
                'average_cycle_time_seconds': avg_cycle_time,
                'recent_average_cycle_time_seconds': recent_avg_cycle_time,
                'performance_improvement_percent': improvement,
                'recent_performance_improvement_percent': recent_improvement,
                'target_achievement_percent': target_achievement,
                
                # Quality metrics
                'success_rate_percent': success_rate * 100,
                'comprehensive_analysis_rate_percent': comprehensive_rate * 100,
                'analysis_consistency_avg': consistency_avg,
                'full_validation_rate_percent': sum(1 for c in self.cycle_history if c.full_validation_completed) / total_cycles * 100 if total_cycles > 0 else 100.0,
                'parallel_efficiency_avg': avg_parallel_efficiency,
                
                # Volume metrics
                'cycles_analyzed': total_cycles,
                'successful_cycles': successful_cycles,
                'throughput_cycles_per_hour': throughput,
                
                # Detailed phase performance
                'phase_performance': phase_performance,
                
                # Targets and baselines
                'baseline_cycle_time_seconds': self.baseline_cycle_time,
                'target_cycle_time_seconds': self.target_cycle_time,
                
                # Recent performance (last 10 cycles)
                'recent_cycles': [
                    {
                        'cycle_id': cycle.cycle_id,
                        'duration_seconds': cycle.total_duration_seconds,
                        'success': cycle.success,
                        'agents_processed': cycle.agents_processed,
                        'indicators_calculated': cycle.indicators_calculated,
                        'comprehensive_analysis': getattr(cycle, 'comprehensive_analysis_applied', True),
                        'analysis_consistency': getattr(cycle, 'analysis_consistency_score', 1.0),
                        'parallel_efficiency': cycle.parallel_efficiency,
                        'signal_generated': cycle.signal_generated
                    }
                    for cycle in self.cycle_history[-10:]
                ]
            }

        except Exception as e:
            logger.error(f"Failed to generate performance report: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def get_optimization_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of different optimization techniques."""
        try:
            if len(self.cycle_history) < 10:
                return {'status': 'insufficient_data', 'message': 'Need at least 10 cycles for analysis'}

            # Group cycles by optimization flags
            optimization_analysis = {}
            
            optimization_flags = [
                'enable_parallel_analysis',
                'enable_phase_merging',
                'comprehensive_analysis_enabled'  # NEW
            ]

            for flag in optimization_flags:
                with_optimization = []
                without_optimization = []
                
                for cycle in self.cycle_history:
                    if cycle.success:  # Only analyze successful cycles
                        if cycle.optimization_flags.get(flag, False):
                            with_optimization.append(cycle.total_duration_seconds)
                        else:
                            without_optimization.append(cycle.total_duration_seconds)
                
                if with_optimization and without_optimization:
                    avg_with = statistics.mean(with_optimization)
                    avg_without = statistics.mean(without_optimization)
                    improvement = ((avg_without - avg_with) / avg_without) * 100
                    
                    optimization_analysis[flag] = {
                        'average_with_optimization': avg_with,
                        'average_without_optimization': avg_without,
                        'improvement_percent': improvement,
                        'samples_with': len(with_optimization),
                        'samples_without': len(without_optimization)
                    }

            return {
                'status': 'success',
                'optimization_analysis': optimization_analysis,
                'total_cycles_analyzed': len([c for c in self.cycle_history if c.success])
            }

        except Exception as e:
            logger.error(f"Failed to analyze optimization effectiveness: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics."""
        try:
            current_time = datetime.utcnow()
            
            # Current cycle info
            current_cycle_info = {}
            if self.current_cycle:
                duration = (current_time - self.current_cycle.start_time).total_seconds()
                current_cycle_info = {
                    'cycle_id': self.current_cycle.cycle_id,
                    'duration_so_far': duration,
                    'current_phase': self.current_phase.phase_name if self.current_phase else None,
                    'phases_completed': len(self.current_cycle.phases),
                    'agents_processed': self.current_cycle.agents_processed,
                    'comprehensive_analysis': getattr(self.current_cycle, 'comprehensive_analysis_applied', True),
                    'analysis_consistency': getattr(self.current_cycle, 'analysis_consistency_score', 1.0)
                }

            # Recent performance (last hour)
            one_hour_ago = current_time - timedelta(hours=1)
            recent_cycles = [
                c for c in self.cycle_history 
                if c.end_time >= one_hour_ago and c.success
            ]

            recent_stats = {}
            if recent_cycles:
                recent_durations = [c.total_duration_seconds for c in recent_cycles]
                recent_stats = {
                    'cycles_last_hour': len(recent_cycles),
                    'average_duration_last_hour': statistics.mean(recent_durations),
                    'min_duration_last_hour': min(recent_durations),
                    'max_duration_last_hour': max(recent_durations)
                }

            return {
                'status': 'success',
                'timestamp': current_time.isoformat(),
                'current_cycle': current_cycle_info,
                'recent_performance': recent_stats,
                'total_cycles_monitored': len(self.cycle_history),
                'monitoring_active': self.current_cycle is not None
            }

        except Exception as e:
            logger.error(f"Failed to get real-time metrics: {str(e)}")
            return {'status': 'error', 'message': str(e)}