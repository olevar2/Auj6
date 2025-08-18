"""
Hierarchy Manager for the AUJ Platform.

This module manages the hierarchical ranking system of expert agents, tracking their
performance and managing promotions/demotions based on validated, out-of-sample results.
The hierarchy determines which agent becomes the Alpha (primary decision maker) for each
analysis cycle.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from enum import Enum
import asyncio
from collections import defaultdict, deque
import statistics

from ..core.exceptions import HierarchyError, ValidationError
from ..core.data_contracts import (
    AgentRank, AgentPerformanceMetrics, TradeSignal, GradedDeal,
    MarketRegime, ConfidenceLevel
)
from ..core.logging_setup import get_logger
from ..agents.base_agent import BaseAgent

logger = get_logger(__name__)


class RankingCriteria(str, Enum):
    """Criteria for agent ranking."""
    OUT_OF_SAMPLE_PERFORMANCE = "OUT_OF_SAMPLE_PERFORMANCE"
    RISK_ADJUSTED_RETURN = "RISK_ADJUSTED_RETURN"
    RECENT_ACCURACY = "RECENT_ACCURACY"
    CONSISTENCY = "CONSISTENCY"
    REGIME_SPECIALIZATION = "REGIME_SPECIALIZATION"


class PerformanceWindow:
    """Performance tracking window for an agent."""
    
    def __init__(self, window_size: int = 50):
        # Note: This class now requires config_manager parameter in __init__
        # self.config_manager = config_manager or UnifiedConfigManager()
        """
        Initialize performance window.
        
        Args:
            window_size: Maximum number of recent trades to track
        """
        self.window_size = window_size
        self.trades: deque = deque(maxlen=window_size)
        self.out_of_sample_trades: deque = deque(maxlen=window_size)
        self.last_updated = datetime.utcnow()
    
    def add_trade(self, trade: GradedDeal, is_out_of_sample: bool = False):
        """Add a completed trade to the performance window."""
        self.trades.append(trade)
        if is_out_of_sample:
            self.out_of_sample_trades.append(trade)
        self.last_updated = datetime.utcnow()
    
    def get_win_rate(self, out_of_sample_only: bool = False) -> float:
        """Calculate win rate for the performance window."""
        trades = self.out_of_sample_trades if out_of_sample_only else self.trades
        if not trades:
            return 0.0
        
        winning_trades = sum(1 for trade in trades if trade.pnl and trade.pnl > 0)
        return winning_trades / len(trades)
    
    def get_total_pnl(self, out_of_sample_only: bool = False) -> Decimal:
        """Calculate total PnL for the performance window."""
        trades = self.out_of_sample_trades if out_of_sample_only else self.trades
        return sum(trade.pnl or Decimal('0') for trade in trades)
    
    def get_sharpe_ratio(self, out_of_sample_only: bool = False) -> Optional[float]:
        """Calculate Sharpe ratio for the performance window."""
        trades = self.out_of_sample_trades if out_of_sample_only else self.trades
        if len(trades) < 10:  # Need sufficient data
            return None
        
        returns = [float(trade.pnl_percentage or 0) for trade in trades if trade.pnl_percentage]
        if not returns:
            return None
        
        mean_return = statistics.mean(returns)
        if len(returns) < 2:
            return None
        
        std_return = statistics.stdev(returns)
        if std_return == 0:
            return None
        
        return mean_return / std_return
    
    def get_max_drawdown(self, out_of_sample_only: bool = False) -> Optional[Decimal]:
        """Calculate maximum drawdown for the performance window."""
        trades = self.out_of_sample_trades if out_of_sample_only else self.trades
        if not trades:
            return None
        
        cumulative_pnl = Decimal('0')
        peak = Decimal('0')
        max_drawdown = Decimal('0')
        
        for trade in trades:
            cumulative_pnl += trade.pnl or Decimal('0')
            peak = max(peak, cumulative_pnl)
            drawdown = peak - cumulative_pnl
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown


class HierarchyManager:
    """
    Manages the hierarchical ranking system of expert agents.
    
    The hierarchy system promotes/demotes agents based on validated performance,
    with emphasis on out-of-sample results to prevent overfitting.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the hierarchy manager.
        
        Args:
            config: Configuration parameters for hierarchy management
        """
        from ..core.unified_config import get_unified_config
        self.config_manager = get_unified_config()
        self.config = config or {}
        
        # Core tracking structures
        self.agents: Dict[str, BaseAgent] = {}
        self.performance_windows: Dict[str, PerformanceWindow] = {}
        self.agent_rankings: Dict[str, float] = {}
        self.regime_specialists: Dict[MarketRegime, List[str]] = defaultdict(list)
        
        # Hierarchy configuration
        self.min_trades_for_ranking = self.config_manager.get_int('min_trades_for_ranking', 20)
        self.out_of_sample_weight = self.config_manager.get_float('out_of_sample_weight', 0.7)
        self.recent_performance_weight = self.config_manager.get_float('recent_performance_weight', 0.6)
        self.consistency_weight = self.config_manager.get_float('consistency_weight', 0.3)
        self.regime_bonus = self.config_manager.get_float('regime_bonus', 0.2)
        
        # Performance decay for older results
        self.performance_decay_days = self.config_manager.get_int('performance_decay_days', 30)
        self.decay_factor = self.config_manager.get_float('decay_factor', 0.95)
        
        # Ranking thresholds
        self.alpha_threshold = self.config_manager.get_float('alpha_threshold', 0.8)
        self.beta_threshold = self.config_manager.get_float('beta_threshold', 0.6)
        self.gamma_threshold = self.config_manager.get_float('gamma_threshold', 0.4)
        
        # Current hierarchy state
        self.current_alpha: Optional[str] = None
        self.current_betas: List[str] = []
        self.current_gammas: List[str] = []
        self.inactive_agents: List[str] = []
        
        self.last_ranking_update = datetime.utcnow()
        
        logger.info("HierarchyManager initialized with anti-overfitting focus")
    
    def register_agent(self, agent: BaseAgent):
        """
        Register an agent with the hierarchy manager.
        
        Args:
            agent: Agent to register
        """
        if agent.name in self.agents:
            logger.warning(f"Agent {agent.name} already registered, updating...")
        
        self.agents[agent.name] = agent
        self.performance_windows[agent.name] = PerformanceWindow()
        self.agent_rankings[agent.name] = 0.0
        
        # Start at lowest rank
        agent.update_rank(AgentRank.GAMMA)
        self.current_gammas.append(agent.name)
        
        logger.info(f"Registered agent: {agent.name} ({agent.specialization})")
    
    def record_trade_result(self, 
                          agent_name: str, 
                          trade: GradedDeal, 
                          is_out_of_sample: bool = False):
        """
        Record a completed trade result for an agent.
        
        Args:
            agent_name: Name of the agent
            trade: Completed trade with results
            is_out_of_sample: Whether this trade was from out-of-sample validation
        """
        if agent_name not in self.agents:
            raise HierarchyError(f"Agent {agent_name} not registered")
        
        # Add to performance window
        self.performance_windows[agent_name].add_trade(trade, is_out_of_sample)
        
        # Update agent's signal success tracking
        agent = self.agents[agent_name]
        if trade.pnl and trade.pnl > 0:
            agent.successful_signals += 1
        
        # Log the result with out-of-sample indicator
        sample_type = "out-of-sample" if is_out_of_sample else "in-sample"
        logger.info(f"Recorded {sample_type} trade for {agent_name}: "
                   f"PnL={trade.pnl}, Grade={trade.grade}")
        
        # Trigger ranking update if enough new data
        self._check_for_ranking_update()
    
    def update_agent_rankings(self, current_regime: Optional[MarketRegime] = None):
        """
        Update agent rankings based on comprehensive performance metrics.
        
        Args:
            current_regime: Current market regime for regime-specific bonuses
        """
        logger.info("Updating agent rankings based on validated performance...")
        
        new_rankings = {}
        
        for agent_name, agent in self.agents.items():
            ranking_score = self._calculate_comprehensive_ranking(agent_name, current_regime)
            new_rankings[agent_name] = ranking_score
        
        # Update rankings
        self.agent_rankings = new_rankings
        
        # Reassign hierarchy positions
        self._reassign_hierarchy_positions()
        
        self.last_ranking_update = datetime.utcnow()
        
        logger.info("Agent rankings updated successfully")
        self._log_current_hierarchy()
    
    def _calculate_comprehensive_ranking(self, 
                                       agent_name: str, 
                                       current_regime: Optional[MarketRegime] = None) -> float:
        """
        Calculate comprehensive ranking score for an agent.
        
        This method heavily weights out-of-sample performance to prevent overfitting.
        """
        agent = self.agents[agent_name]
        window = self.performance_windows[agent_name]
        
        # Insufficient data for ranking
        if len(window.trades) < self.min_trades_for_ranking:
            return 0.0
        
        # Base scores
        base_score = 0.0
        
        # 1. Out-of-sample performance (highest weight)
        oos_win_rate = window.get_win_rate(out_of_sample_only=True)
        oos_sharpe = window.get_sharpe_ratio(out_of_sample_only=True) or 0.0
        oos_pnl = float(window.get_total_pnl(out_of_sample_only=True))
        
        if len(window.out_of_sample_trades) >= 10:  # Need sufficient OOS data
            oos_score = (oos_win_rate * 0.4 + 
                        min(oos_sharpe / 2.0, 1.0) * 0.3 + 
                        min(oos_pnl / 1000.0, 1.0) * 0.3)
            base_score += oos_score * self.out_of_sample_weight
        else:
            # Penalty for insufficient out-of-sample data
            base_score *= 0.5
        
        # 2. Recent performance (medium weight)
        recent_win_rate = window.get_win_rate(out_of_sample_only=False)
        recent_sharpe = window.get_sharpe_ratio(out_of_sample_only=False) or 0.0
        
        recent_score = (recent_win_rate * 0.6 + 
                       min(recent_sharpe / 2.0, 1.0) * 0.4)
        base_score += recent_score * self.recent_performance_weight
        
        # 3. Consistency bonus
        consistency_score = self._calculate_consistency_score(window)
        base_score += consistency_score * self.consistency_weight
        
        # 4. Regime specialization bonus
        if current_regime and agent_name in self.regime_specialists.get(current_regime, []):
            base_score += self.regime_bonus
        
        # 5. Anti-overfitting penalty
        overfitting_penalty = self._calculate_overfitting_penalty(window)
        base_score *= (1.0 - overfitting_penalty)
        
        # 6. Time decay for older performance
        age_factor = self._calculate_age_factor(window.last_updated)
        base_score *= age_factor
        
        return max(0.0, min(1.0, base_score))
    
    def _calculate_consistency_score(self, window: PerformanceWindow) -> float:
        """Calculate consistency score based on performance stability."""
        if len(window.trades) < 10:
            return 0.0
        
        # Calculate rolling win rates
        rolling_window = 10
        rolling_win_rates = []
        
        trades_list = list(window.trades)
        for i in range(rolling_window, len(trades_list) + 1):
            recent_trades = trades_list[i-rolling_window:i]
            wins = sum(1 for trade in recent_trades if trade.pnl and trade.pnl > 0)
            rolling_win_rates.append(wins / rolling_window)
        
        if len(rolling_win_rates) < 3:
            return 0.0
        
        # Lower standard deviation = higher consistency
        consistency = 1.0 - min(statistics.stdev(rolling_win_rates), 1.0)
        return consistency
    
    def _calculate_overfitting_penalty(self, window: PerformanceWindow) -> float:
        """
        Calculate penalty for overfitting based on in-sample vs out-of-sample performance gap.
        """
        if len(window.out_of_sample_trades) < 5 or len(window.trades) < 10:
            return 0.0
        
        in_sample_win_rate = window.get_win_rate(out_of_sample_only=False)
        oos_win_rate = window.get_win_rate(out_of_sample_only=True)
        
        # Large performance gap suggests overfitting
        performance_gap = max(0.0, in_sample_win_rate - oos_win_rate)
        
        # Penalty increases with gap size
        penalty = min(performance_gap * 2.0, 0.5)  # Max 50% penalty
        
        return penalty
    
    def _calculate_age_factor(self, last_updated: datetime) -> float:
        """Calculate age factor for performance decay."""
        days_old = (datetime.utcnow() - last_updated).days
        if days_old <= self.performance_decay_days:
            return 1.0
        
        # Exponential decay after threshold
        excess_days = days_old - self.performance_decay_days
        return self.decay_factor ** excess_days
    
    def _reassign_hierarchy_positions(self):
        """Reassign agents to hierarchy positions based on rankings."""
        # Sort agents by ranking score
        sorted_agents = sorted(self.agent_rankings.items(), 
                             key=lambda x: x[1], 
                             reverse=True)
        
        # Clear current positions
        self.current_alpha = None
        self.current_betas.clear()
        self.current_gammas.clear()
        self.inactive_agents.clear()
        
        # Assign new positions
        for agent_name, score in sorted_agents:
            agent = self.agents[agent_name]
            
            if score >= self.alpha_threshold and self.current_alpha is None:
                # Assign Alpha (only one)
                self.current_alpha = agent_name
                agent.update_rank(AgentRank.ALPHA)
                
            elif score >= self.beta_threshold and len(self.current_betas) < 3:
                # Assign Beta (max 3)
                self.current_betas.append(agent_name)
                agent.update_rank(AgentRank.BETA)
                
            elif score >= self.gamma_threshold:
                # Assign Gamma
                self.current_gammas.append(agent_name)
                agent.update_rank(AgentRank.GAMMA)
                
            else:
                # Inactive due to poor performance
                self.inactive_agents.append(agent_name)
                agent.update_rank(AgentRank.INACTIVE)
    
    def get_alpha_agent(self) -> Optional[BaseAgent]:
        """Get the current Alpha agent."""
        if self.current_alpha:
            return self.agents[self.current_alpha]
        return None
    
    def get_beta_agents(self) -> List[BaseAgent]:
        """Get current Beta agents."""
        return [self.agents[name] for name in self.current_betas]
    
    def get_gamma_agents(self) -> List[BaseAgent]:
        """Get current Gamma agents."""
        return [self.agents[name] for name in self.current_gammas]
    
    def get_agent_by_rank(self, rank: AgentRank) -> List[BaseAgent]:
        """Get agents by specific rank."""
        if rank == AgentRank.ALPHA:
            return [self.get_alpha_agent()] if self.current_alpha else []
        elif rank == AgentRank.BETA:
            return self.get_beta_agents()
        elif rank == AgentRank.GAMMA:
            return self.get_gamma_agents()
        elif rank == AgentRank.INACTIVE:
            return [self.agents[name] for name in self.inactive_agents]
        else:
            return []
    
    def get_regime_specialists(self, regime: MarketRegime) -> List[BaseAgent]:
        """Get agents specialized for a specific market regime."""
        specialist_names = self.regime_specialists.get(regime, [])
        return [self.agents[name] for name in specialist_names if name in self.agents]
    
    def update_regime_specialization(self, 
                                   agent_name: str, 
                                   regime: MarketRegime, 
                                   performance_score: float):
        """
        Update an agent's specialization for a specific market regime.
        
        Args:
            agent_name: Name of the agent
            regime: Market regime
            performance_score: Performance score for this regime (0.0 to 1.0)
        """
        if performance_score >= 0.7:  # High performance threshold
            if agent_name not in self.regime_specialists[regime]:
                self.regime_specialists[regime].append(agent_name)
                logger.info(f"Agent {agent_name} now specializes in {regime.value}")
        else:
            # Remove specialization if performance drops
            if agent_name in self.regime_specialists[regime]:
                self.regime_specialists[regime].remove(agent_name)
                logger.info(f"Agent {agent_name} lost specialization in {regime.value}")
    
    def get_hierarchy_summary(self) -> Dict[str, Any]:
        """Get comprehensive hierarchy summary."""
        return {
            "last_updated": self.last_ranking_update.isoformat(),
            "alpha_agent": {
                "name": self.current_alpha,
                "score": self.agent_rankings.get(self.current_alpha, 0.0) if self.current_alpha else None
            },
            "beta_agents": [
                {
                    "name": name,
                    "score": self.agent_rankings.get(name, 0.0)
                }
                for name in self.current_betas
            ],
            "gamma_agents": [
                {
                    "name": name,
                    "score": self.agent_rankings.get(name, 0.0)
                }
                for name in self.current_gammas
            ],
            "inactive_agents": [
                {
                    "name": name,
                    "score": self.agent_rankings.get(name, 0.0)
                }
                for name in self.inactive_agents
            ],
            "total_agents": len(self.agents),
            "regime_specialists": {
                regime.value: specialist_names
                for regime, specialist_names in self.regime_specialists.items()
            }
        }
    
    def get_agent_performance_details(self, agent_name: str) -> Dict[str, Any]:
        """Get detailed performance metrics for a specific agent."""
        if agent_name not in self.agents:
            raise HierarchyError(f"Agent {agent_name} not found")
        
        agent = self.agents[agent_name]
        window = self.performance_windows[agent_name]
        
        return {
            "agent_name": agent_name,
            "current_rank": agent.current_rank.value,
            "ranking_score": self.agent_rankings.get(agent_name, 0.0),
            "specialization": agent.specialization,
            "total_trades": len(window.trades),
            "out_of_sample_trades": len(window.out_of_sample_trades),
            "overall_performance": {
                "win_rate": window.get_win_rate(out_of_sample_only=False),
                "total_pnl": float(window.get_total_pnl(out_of_sample_only=False)),
                "sharpe_ratio": window.get_sharpe_ratio(out_of_sample_only=False),
                "max_drawdown": float(window.get_max_drawdown(out_of_sample_only=False) or 0)
            },
            "out_of_sample_performance": {
                "win_rate": window.get_win_rate(out_of_sample_only=True),
                "total_pnl": float(window.get_total_pnl(out_of_sample_only=True)),
                "sharpe_ratio": window.get_sharpe_ratio(out_of_sample_only=True),
                "max_drawdown": float(window.get_max_drawdown(out_of_sample_only=True) or 0)
            },
            "consistency_score": self._calculate_consistency_score(window),
            "overfitting_penalty": self._calculate_overfitting_penalty(window),
            "last_updated": window.last_updated.isoformat()
        }
    
    def _check_for_ranking_update(self):
        """Check if rankings should be updated based on new data."""
        # Update rankings every hour or when significant new data arrives
        time_since_update = datetime.utcnow() - self.last_ranking_update
        
        if time_since_update.total_seconds() >= 3600:  # 1 hour
            self.update_agent_rankings()
    
    def _log_current_hierarchy(self):
        """Log current hierarchy state."""
        logger.info("=== Current Agent Hierarchy ===")
        
        if self.current_alpha:
            score = self.agent_rankings.get(self.current_alpha, 0.0)
            logger.info(f"ALPHA: {self.current_alpha} (Score: {score:.3f})")
        
        if self.current_betas:
            logger.info("BETA Agents:")
            for name in self.current_betas:
                score = self.agent_rankings.get(name, 0.0)
                logger.info(f"  - {name} (Score: {score:.3f})")
        
        if self.current_gammas:
            logger.info(f"GAMMA Agents: {len(self.current_gammas)} agents")
        
        if self.inactive_agents:
            logger.info(f"INACTIVE Agents: {len(self.inactive_agents)} agents")
        
        logger.info("==============================")
    
    async def emergency_hierarchy_reset(self):
        """Emergency reset of hierarchy (e.g., if Alpha agent fails consistently)."""
        logger.warning("Performing emergency hierarchy reset...")
        
        # Reset all agents to Gamma
        for agent in self.agents.values():
            agent.update_rank(AgentRank.GAMMA)
        
        # Clear hierarchy
        self.current_alpha = None
        self.current_betas.clear()
        self.current_gammas = list(self.agents.keys())
        self.inactive_agents.clear()
        
        # Force immediate ranking update
        self.update_agent_rankings()
        
        logger.warning("Emergency hierarchy reset completed")
    
    def validate_hierarchy_integrity(self) -> bool:
        """Validate hierarchy integrity and consistency."""
        try:
            # Check that all agents are accounted for
            all_positioned = (
                [self.current_alpha] if self.current_alpha else [] +
                self.current_betas +
                self.current_gammas +
                self.inactive_agents
            )
            
            if set(all_positioned) != set(self.agents.keys()):
                logger.error("Hierarchy integrity check failed: agent mismatch")
                return False
            
            # Check that Alpha has highest score
            if self.current_alpha:
                alpha_score = self.agent_rankings.get(self.current_alpha, 0.0)
                for other_agent, score in self.agent_rankings.items():
                    if other_agent != self.current_alpha and score > alpha_score:
                        logger.warning(f"Hierarchy inconsistency: {other_agent} has higher score than Alpha")
            
            return True
            
        except Exception as e:
            logger.error(f"Hierarchy validation failed: {str(e)}")
            return False