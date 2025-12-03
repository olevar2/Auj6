"""
Hierarchy Manager for the AUJ Platform.

This module manages the hierarchical ranking system of expert agents, tracking their
performance and managing promotions/demotions based on validated, out-of-sample results.
The hierarchy determines which agent becomes the Alpha (primary decision maker) for each
analysis cycle.

FIXES IMPLEMENTED:
- Fixed syntax error in line 478: proper list concatenation with parentheses
- Added missing integration methods: initialize(), register_agent(), get_current_agent_rankings(), update_agent_rank(), save_agent_rankings()
- Fixed broken __init__ method structure (lines 133-156)
- All platform integration points restored
- ✅ BUG #30 FIX: Implemented proper database persistence for agent rankings
- ✅ Fixed initialize() method to load rankings from database
- ✅ Fixed save_agent_rankings() method to actually persist data to database
- ✅ Added _ensure_rankings_table_exists() for database schema creation
- ✅ Added _load_rankings_from_database() for data restoration
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from enum import Enum
import asyncio
from collections import defaultdict, deque
import statistics
import json

from ..core.exceptions import HierarchyError, ValidationError
from ..core.data_contracts import (
    AgentRank, AgentPerformanceMetrics, TradeSignal, GradedDeal,
    MarketRegime, ConfidenceLevel
)
from ..core.logging_setup import get_logger
from ..agents.base_agent import BaseAgent
from ..core.unified_database_manager import get_unified_database

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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, database=None):
        """
        Initialize the hierarchy manager.
        
        Args:
            config: Configuration parameters for hierarchy management
            database: Database manager instance (injected for dependency injection)
        """
        from ..core.unified_config import get_unified_config
        self.config_manager = get_unified_config()
        self.config = config or {}
        
        # ✅ BUG #30 FIX: Add database manager with dependency injection support
        self.database = database or get_unified_database()
        
        self.agents = {}
        self.performance_windows = {}
        self.regime_specialists = defaultdict(list)
        self.agent_rankings = {}
        
        self.current_alpha = None
        self.current_betas = []
        self.current_gammas = []
        self.inactive_agents = []
        
        self.last_ranking_update = datetime.utcnow()
        
        # Config parameters
        self.min_trades_for_ranking = self.config.get('min_trades', 10)
        self.out_of_sample_weight = self.config.get('oos_weight', 0.4)
        self.recent_performance_weight = self.config.get('recent_weight', 0.3)
        self.consistency_weight = self.config.get('consistency_weight', 0.2)
        self.regime_bonus = self.config.get('regime_bonus', 0.1)
        self.performance_decay_days = self.config.get('decay_days', 30)
        self.decay_factor = self.config.get('decay_factor', 0.95)
        
        self.alpha_threshold = self.config.get('alpha_threshold', 0.8)
        self.beta_threshold = self.config.get('beta_threshold', 0.6)
        self.gamma_threshold = self.config.get('gamma_threshold', 0.4)
        
        self._ranking_lock = asyncio.Lock()
        
        logger.info("Hierarchy Manager initialized")
    
    async def _ensure_rankings_table_exists(self):
        """
        Create agent_rankings table if it doesn't exist.
        
        ✅ BUG #30 FIX: New method to ensure database schema exists
        """
        try:
            create_table_query = """
            CREATE TABLE IF NOT EXISTS agent_rankings (
                id INTEGER PRIMARY KEY DEFAULT 1,
                rankings_data TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                CHECK (id = 1)
            )
            """
            await self.database.execute_query(create_table_query)
            logger.debug("Agent rankings table schema verified/created")
        except Exception as e:
            logger.error(f"Failed to create agent_rankings table: {e}")
            raise HierarchyError(f"Database schema creation failed: {str(e)}")
    
    async def _load_rankings_from_database(self) -> bool:
        """
        Load agent rankings from database.
        
        ✅ BUG #30 FIX: New method to restore rankings from persistent storage
        
        Returns:
            True if data was loaded successfully, False if no data exists
        """
        try:
            result = await self.database.fetch_one(
                "SELECT rankings_data, last_updated FROM agent_rankings WHERE id = 1"
            )
            
            if not result:
                logger.debug("No saved rankings found in database")
                return False
            
            # Parse JSON data
            data = json.loads(result['rankings_data'])
            
            # Restore agent rankings
            self.agent_rankings = {k: float(v) for k, v in data.get('rankings', {}).items()}
            
            # Restore hierarchy positions
            self.current_alpha = data.get('alpha')
            self.current_betas = data.get('betas', [])
            self.current_gammas = data.get('gammas', [])
            self.inactive_agents = data.get('inactive', [])
            
            # Restore regime specialists
            regime_data = data.get('regime_specialists', {})
            for regime_str, agent_list in regime_data.items():
                try:
                    regime = MarketRegime(regime_str)
                    self.regime_specialists[regime] = agent_list
                except ValueError:
                    logger.warning(f"Unknown market regime '{regime_str}' in saved data, skipping")
            
            # Restore last update timestamp
            self.last_ranking_update = datetime.fromisoformat(data.get('last_updated'))
            
            logger.info(f"Successfully loaded {len(self.agent_rankings)} agent rankings from database")
            logger.debug(f"Hierarchy: Alpha={self.current_alpha}, Betas={len(self.current_betas)}, "
                        f"Gammas={len(self.current_gammas)}, Inactive={len(self.inactive_agents)}")
            
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse rankings JSON data: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load rankings from database: {e}")
            return False
    
    async def initialize(self):
        """
        Initialize the hierarchy manager and load saved data.
        
        ✅ BUG #30 FIX: Properly implemented to load rankings from database
        """
        logger.info("Initializing Hierarchy Manager...")
        
        try:
            # Ensure database table exists
            await self._ensure_rankings_table_exists()
            
            # Load saved rankings from database
            loaded = await self._load_rankings_from_database()
            
            if loaded:
                logger.info(f"✅ Loaded {len(self.agent_rankings)} agent rankings from database")
                # Log current hierarchy state
                self._log_current_hierarchy()
            else:
                logger.info("No saved rankings found, starting with fresh hierarchy")
            
        except Exception as e:
            logger.warning(f"Could not load saved rankings: {e}")
            logger.info("Continuing with fresh hierarchy initialization")
        
        logger.info("Hierarchy Manager initialization complete")
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the hierarchy manager."""
        if agent.name in self.agents:
            logger.warning(f"Agent {agent.name} already registered, updating...")
        
        self.agents[agent.name] = agent
        self.performance_windows[agent.name] = PerformanceWindow()
        
        # Only initialize ranking if not already loaded from database
        if agent.name not in self.agent_rankings:
            self.agent_rankings[agent.name] = 0.0
        
        logger.info(f"Registered agent: {agent.name}")

    async def update_agent_rankings(self, current_regime: Optional[MarketRegime] = None):
        """
        Update agent rankings based on validated performance.
        
        Args:
            current_regime: Current market regime for regime-specific bonuses
        """
        async with self._ranking_lock:
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
            
            # ✅ BUG #30 FIX: Auto-save rankings after update
            try:
                await self.save_agent_rankings()
            except Exception as e:
                logger.error(f"Failed to auto-save rankings after update: {e}")

    async def record_trade_result(self, agent_name: str, trade: GradedDeal, is_out_of_sample: bool = False):
        """
        Record a completed trade result for an agent.
        
        Args:
            agent_name: Name of the agent
            trade: The completed trade details
            is_out_of_sample: Whether this is an out-of-sample trade
        """
        async with self._ranking_lock:
            if agent_name not in self.agents:
                raise HierarchyError(f"Agent {agent_name} not registered")
            
            # Update performance window
            self.performance_windows[agent_name].add_trade(trade, is_out_of_sample)
            
            # Update agent stats
            agent = self.agents[agent_name]
            if trade.pnl and trade.pnl > 0:
                agent.successful_signals += 1
            
            sample_type = "out-of-sample" if is_out_of_sample else "in-sample"
            logger.info(f"Recorded {sample_type} trade for {agent_name}: PnL={trade.pnl}, Grade={trade.grade}")
            
            # Check if rankings need update
            await self._check_for_ranking_update()
    
    async def get_current_agent_rankings(self) -> Dict[str, Any]:
        """Get current agent rankings and hierarchy positions."""
        return {
            "rankings": self.agent_rankings.copy(),
            "alpha": self.current_alpha,
            "betas": self.current_betas.copy(),
            "gammas": self.current_gammas.copy(),
            "inactive": self.inactive_agents.copy(),
            "last_updated": self.last_ranking_update.isoformat()
        }
    
    async def update_agent_rank(self, agent_name: str, new_rank: AgentRank):
        """Manually update an agent's rank."""
        if agent_name not in self.agents:
            raise HierarchyError(f"Agent {agent_name} not registered")
        
        agent = self.agents[agent_name]
        old_rank = agent.current_rank
        
        # Remove from old position
        if self.current_alpha == agent_name:
            self.current_alpha = None
        if agent_name in self.current_betas:
            self.current_betas.remove(agent_name)
        if agent_name in self.current_gammas:
            self.current_gammas.remove(agent_name)
        if agent_name in self.inactive_agents:
            self.inactive_agents.remove(agent_name)
        
        # Add to new position
        if new_rank == AgentRank.ALPHA:
            if self.current_alpha:
                logger.warning(f"Replacing current alpha {self.current_alpha} with {agent_name}")
            self.current_alpha = agent_name
        elif new_rank == AgentRank.BETA:
            self.current_betas.append(agent_name)
        elif new_rank == AgentRank.GAMMA:
            self.current_gammas.append(agent_name)
        else:  # INACTIVE
            self.inactive_agents.append(agent_name)
        
        agent.update_rank(new_rank)
        logger.info(f"Updated {agent_name} rank: {old_rank.value} → {new_rank.value}")
        
        # ✅ BUG #30 FIX: Save rankings after manual update
        try:
            await self.save_agent_rankings()
        except Exception as e:
            logger.error(f"Failed to save rankings after manual rank update: {e}")
    
    async def save_agent_rankings(self):
        """
        Save current agent rankings to persistent storage.
        
        ✅ BUG #30 FIX: Properly implemented to actually save to database
        """
        try:
            # Prepare rankings data
            rankings_data = {
                "rankings": {k: float(v) for k, v in self.agent_rankings.items()},
                "alpha": self.current_alpha,
                "betas": self.current_betas.copy(),
                "gammas": self.current_gammas.copy(),
                "inactive": self.inactive_agents.copy(),
                "last_updated": self.last_ranking_update.isoformat(),
                "regime_specialists": {k.value: v for k, v in self.regime_specialists.items()}
            }
            
            # Serialize to JSON
            json_data = json.dumps(rankings_data)
            current_time = datetime.utcnow().isoformat()
            
            # Save to database using INSERT OR REPLACE (works for SQLite)
            # For PostgreSQL this would use INSERT ... ON CONFLICT, but the database
            # manager handles the differences internally
            await self.database.execute_query("""
                INSERT OR REPLACE INTO agent_rankings (id, rankings_data, last_updated)
                VALUES (1, ?, ?)
            """, (json_data, current_time))
            
            logger.info(f"✅ Agent rankings saved successfully ({len(self.agent_rankings)} agents)")
            logger.debug(f"Saved hierarchy: Alpha={self.current_alpha}, "
                        f"Betas={len(self.current_betas)}, Gammas={len(self.current_gammas)}")
            
            return rankings_data
            
        except Exception as e:
            logger.error(f"Failed to save agent rankings: {e}")
            raise HierarchyError(f"Rankings persistence failed: {str(e)}")
    
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
    
    async def _check_for_ranking_update(self):
        """Check if rankings should be updated based on new data."""
        # Update rankings every hour or when significant new data arrives
        time_since_update = datetime.utcnow() - self.last_ranking_update
        
        if time_since_update.total_seconds() >= 3600:  # 1 hour
            await self.update_agent_rankings()
    
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
        await self.update_agent_rankings()
        
        logger.warning("Emergency hierarchy reset completed")
    
    def validate_hierarchy_integrity(self) -> bool:
        """Validate hierarchy integrity and consistency."""
        try:
            # Check that all agents are accounted for
            # FIXED: Added parentheses around ternary operator for proper list concatenation
            all_positioned = (
                ([self.current_alpha] if self.current_alpha else []) +
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
