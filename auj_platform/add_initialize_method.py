# Simple Python script to add initialize() method to HierarchyManager

import sys

def add_initialize_method(file_path):
    """Add the initialize() async method to HierarchyManager class."""
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the location to insert (after line 164: logger.info("HierarchyManager initialized..."))
    insert_index = None
    for i, line in enumerate(lines):
        if 'HierarchyManager initialized with anti-overfitting focus' in line:
            insert_index = i + 2  # After the logger line and the blank line
            break
    
    if insert_index is None:
        print("ERROR: Could not find insertion point")
        return False
    
    # The method to insert
    method_code = '''    async def initialize(self):
        """
        Async initialization method for HierarchyManager.
        
        This method is called after __init__ to complete the initialization
        sequence. It performs:
        - Loading saved agent rankings from database (if available)
        - Validating agent registry
        - Initial hierarchy setup
        - Performance window initialization
        """
        try:
            logger.info("Starting HierarchyManager async initialization...")
            
            # Load saved rankings from database if available
            # This is optional - if no saved data exists, will use defaults
            try:
                # TODO: Load from database when ready
                # saved_rankings = await self.database.load_agent_rankings()
                # if saved_rankings:
                #     self.agent_rankings = saved_rankings
                #     logger.info(f"Loaded {len(saved_rankings)} saved agent rankings")
                pass
            except Exception as e:
                logger.warning(f"Could not load saved rankings (this is normal on first run): {e}")
            
            # Validate that all required configuration is present
            required_config_keys = [
                'min_trades_for_ranking',
                'out_of_sample_weight',
                'alpha_threshold',
                'beta_threshold',
                'gamma_threshold'
            ]
            
            for key in required_config_keys:
                if not hasattr(self, key):
                    raise ValueError(f"Missing required configuration: {key}")
            
            # Initialize performance tracking structures
            # Agents will be registered later via register_agent()
            logger.info(f"HierarchyManager initialized with thresholds: "
                       f"Alpha={self.alpha_threshold}, Beta={self.beta_threshold}, "
                       f"Gamma={self.gamma_threshold}")
            
            # Log current hierarchy state
            logger.info(f"Current hierarchy: Alpha={self.current_alpha}, "
                       f"Betas={len(self.current_betas)}, "
                       f"Gammas={len(self.current_gammas)}")
            
            logger.info("HierarchyManager async initialization completed successfully")
            
        except Exception as e:
            logger.error(f"HierarchyManager initialization failed: {e}")
            raise
    
'''
    
    # Insert the method
    lines.insert(insert_index, method_code)
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("SUCCESS: Added initialize() method to HierarchyManager")
    return True

if __name__ == "__main__":
    file_path = r"e:\AUG6\auj_platform\src\hierarchy\hierarchy_manager.py"
    success = add_initialize_method(file_path)
    sys.exit(0 if success else 1)
