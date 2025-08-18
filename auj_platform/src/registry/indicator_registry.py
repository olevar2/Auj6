# Indicator registry

class IndicatorRegistry:
    """Simple indicator registry for dashboard compatibility"""

    def __init__(self):
        self.indicators = {}

    def register_indicator(self, name, indicator_class):
        """Register an indicator"""
        self.indicators[name] = indicator_class

    def get_indicator(self, name):
        """Get an indicator by name"""
        return self.indicators.get(name)

    def list_indicators(self):
        """List all registered indicators"""
        return list(self.indicators.keys())
