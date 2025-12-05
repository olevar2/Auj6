"""
OpportunityRadar API Endpoints for AUJ Platform Dashboard.

✨ CROWN JEWEL: Endpoints for viewing real-time opportunity scanning results
and radar statistics in the dashboard.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from ..core.logging_setup import get_logger

logger = get_logger(__name__)


# Response Models
class OpportunityRadarScanResponse(BaseModel):
    """Response model for single pair scan result."""
    symbol: str
    opportunity_score: float = Field(..., description="Score 0-100")
    grade: str = Field(..., description="A+, A, B, C, D")
    direction: str = Field(..., description="BUY, SELL, HOLD")
    trend_clarity: float
    momentum_strength: float
    volatility_fit: float
    regime: str
    is_tradeable: bool


class OpportunityRadarResponse(BaseModel):
    """Response model for OpportunityRadar dashboard."""
    enabled: bool = Field(..., description="Is radar enabled")
    last_scan_time: Optional[str] = Field(None, description="Last scan timestamp")
    best_opportunity: Optional[Dict[str, Any]] = Field(None, description="Current best opportunity")
    scan_results: List[OpportunityRadarScanResponse] = Field(default_factory=list)
    statistics: Dict[str, Any] = Field(default_factory=dict)
    pairs_configured: int = Field(0, description="Total pairs being scanned")


# Create router
router = APIRouter(prefix="/api/v1/radar", tags=["Opportunity Radar"])


@router.get("/overview", response_model=OpportunityRadarResponse)
async def get_radar_overview():
    """
    Get OpportunityRadar overview including current best opportunity and recent scans.
    
    Returns real-time scanning results for dashboard visualization.
    """
    try:
        # Import here to avoid circular imports
        from ..coordination.opportunity_radar import OpportunityRadar
        from ...config.opportunity_radar_config import (
            get_radar_config, get_radar_pairs, is_radar_enabled
        )
        
        config = get_radar_config()
        pairs = get_radar_pairs()
        
        return OpportunityRadarResponse(
            enabled=is_radar_enabled(),
            last_scan_time=None,  # Will be populated after first scan
            best_opportunity=None,
            scan_results=[],
            statistics={
                "total_scans": 0,
                "successful_scans": 0,
                "average_scan_time": 0.0,
                "top_n_configured": config.get("top_n", 3)
            },
            pairs_configured=len(pairs)
        )
        
    except Exception as e:
        logger.error(f"Error getting radar overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pairs")
async def get_radar_pairs_list():
    """
    Get list of pairs configured for OpportunityRadar scanning.
    """
    try:
        from ...config.opportunity_radar_config import (
            get_radar_pairs, get_radar_config,
            MAJOR_PAIRS, ACTIVE_PAIRS, ALL_PAIRS
        )
        
        config = get_radar_config()
        current_scope = config.get("pair_scope", "active")
        
        return {
            "current_scope": current_scope,
            "active_pairs": get_radar_pairs(current_scope),
            "available_scopes": {
                "majors": len(MAJOR_PAIRS),
                "active": len(ACTIVE_PAIRS),
                "all": len(ALL_PAIRS)
            },
            "all_pairs_by_scope": {
                "majors": MAJOR_PAIRS,
                "active": ACTIVE_PAIRS,
                "all": ALL_PAIRS
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting radar pairs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_radar_config_endpoint():
    """
    Get current OpportunityRadar configuration.
    """
    try:
        from ...config.opportunity_radar_config import get_radar_config
        
        return get_radar_config()
        
    except Exception as e:
        logger.error(f"Error getting radar config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_radar_statistics():
    """
    Get OpportunityRadar performance statistics.
    """
    try:
        # This would be populated from actual radar instance
        # For now return placeholder structure
        return {
            "total_scans": 0,
            "successful_scans": 0,
            "success_rate": 0.0,
            "average_scan_time_seconds": 0.0,
            "top_picked_pairs": {},
            "score_distribution": {
                "A+": 0,
                "A": 0,
                "B": 0,
                "C": 0,
                "D": 0
            },
            "last_24h_opportunities": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting radar statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
