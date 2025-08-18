"""
AUJ Platform Path Cleanup Utility
=================================
Minimal, non-invasive path management and import cleanup for the AUJ Platform.
This utility fixes import paths and references without disrupting existing architecture.

Author: AUJ Platform Development Team  
Date: 2025-07-04
Version: 1.0.0
"""
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import ast
import importlib.util


class PathCleanupUtility:
    """
    Utility for cleaning up import paths and references in the AUJ Platform
    with minimal impact on existing architecture.
    """
    
    def __init__(self, platform_root: Optional[str] = None):
        """Initialize path cleanup utility"""
        self.platform_root = Path(platform_root) if platform_root else Path(__file__).parent.parent.parent
        self.src_path = self.platform_root / "auj_platform" / "src"
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Track changes made
        self.changes_made = {
            'files_analyzed': 0,
            'imports_fixed': 0,
            'paths_normalized': 0,
            'references_updated': 0,
            'files_modified': []
        }
        
        # Common import patterns that need fixing
        self.import_patterns = {
            # Old pattern -> New pattern
            r'from \.\.coordination\.': 'from coordination.',
            r'from \.\.agents\.': 'from agents.',
            r'from \.\.core\.': 'from core.',
            r'from \.\.trading_engine\.': 'from trading_engine.',
            r'from \.\.broker_interfaces\.': 'from broker_interfaces.',
            r'from \.\.data_providers\.': 'from data_providers.',
            r'from \.\.analytics\.': 'from analytics.',
            r'from \.\.monitoring\.': 'from monitoring.',
            r'from \.\.api\.': 'from api.',
        }
    
    def analyze_import_structure(self) -> Dict[str, any]:
        """
        Analyze the current import structure across the platform.
        Returns analysis results without making changes.
        """
        self.logger.info("Analyzing import structure across the platform...")
        
        analysis_results = {
            'total_files': 0,
            'python_files': 0,
            'import_issues': [],
            'circular_dependencies': [],
            'missing_imports': [],
            'relative_imports': [],
            'absolute_imports': [],
            'problematic_patterns': {}
        }
        
        # Scan all Python files
        for file_path in self.src_path.rglob("*.py"):
            if self._should_analyze_file(file_path):
                analysis_results['total_files'] += 1
                if file_path.suffix == '.py':
                    analysis_results['python_files'] += 1
                    file_analysis = self._analyze_file_imports(file_path)
                    
                    # Aggregate results
                    analysis_results['import_issues'].extend(file_analysis['issues'])
                    analysis_results['relative_imports'].extend(file_analysis['relative_imports'])
                    analysis_results['absolute_imports'].extend(file_analysis['absolute_imports'])
                    
                    # Track problematic patterns
                    for pattern in file_analysis['problematic_patterns']:
                        if pattern not in analysis_results['problematic_patterns']:
                            analysis_results['problematic_patterns'][pattern] = []
                        analysis_results['problematic_patterns'][pattern].append(str(file_path))
        
        self.logger.info(f"Analysis complete: {analysis_results['python_files']} Python files analyzed")
        return analysis_results
    
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Determine if a file should be analyzed"""
        # Skip certain directories and files
        skip_patterns = [
            '__pycache__',
            '.git',
            '.pytest_cache',
            'test_',
            '_test.py',
            '.backup',
            '.bak'
        ]
        
        file_str = str(file_path)
        return not any(pattern in file_str for pattern in skip_patterns)
    
    def _analyze_file_imports(self, file_path: Path) -> Dict[str, any]:
        """Analyze imports in a specific file"""
        file_analysis = {
            'file': str(file_path),
            'issues': [],
            'relative_imports': [],
            'absolute_imports': [],
            'problematic_patterns': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST for more accurate analysis
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            file_analysis['absolute_imports'].append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ''
                        if node.level > 0:  # Relative import
                            relative_import = '.' * node.level + module
                            file_analysis['relative_imports'].append(relative_import)
                        else:
                            file_analysis['absolute_imports'].append(module)
            except SyntaxError:
                # If AST parsing fails, fall back to regex analysis
                self.logger.warning(f"Syntax error in {file_path}, using regex analysis")
                self._analyze_imports_with_regex(content, file_analysis)
            
            # Check for problematic patterns
            for pattern in self.import_patterns.keys():
                if re.search(pattern, content):
                    file_analysis['problematic_patterns'].append(pattern)
            
        except Exception as e:
            file_analysis['issues'].append(f"Error reading file: {str(e)}")
        
        return file_analysis
    
    def _analyze_imports_with_regex(self, content: str, file_analysis: Dict):
        """Fallback regex-based import analysis"""
        # Find import statements
        import_patterns = [
            r'from\s+([\w\.]+)\s+import',
            r'import\s+([\w\.]+)'
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if match.startswith('.'):
                    file_analysis['relative_imports'].append(match)
                else:
                    file_analysis['absolute_imports'].append(match)
    
    def fix_import_paths(self, dry_run: bool = True) -> Dict[str, any]:
        """
        Fix import paths across the platform.
        
        Args:
            dry_run: If True, only analyze without making changes
            
        Returns:
            Results of the fixing process
        """
        self.logger.info(f"{'Analyzing' if dry_run else 'Fixing'} import paths...")
        
        fix_results = {
            'files_processed': 0,
            'files_modified': 0,
            'imports_fixed': 0,
            'changes': [],
            'errors': []
        }
        
        # Process all Python files
        for file_path in self.src_path.rglob("*.py"):
            if self._should_analyze_file(file_path):
                fix_results['files_processed'] += 1
                
                try:
                    file_changes = self._fix_file_imports(file_path, dry_run)
                    if file_changes['modified']:
                        fix_results['files_modified'] += 1
                        fix_results['imports_fixed'] += file_changes['fixes_made']
                        fix_results['changes'].extend(file_changes['changes'])
                        
                        if not dry_run:
                            self.changes_made['files_modified'].append(str(file_path))
                
                except Exception as e:
                    error_msg = f"Error processing {file_path}: {str(e)}"
                    fix_results['errors'].append(error_msg)
                    self.logger.error(error_msg)
        
        # Update global statistics
        if not dry_run:
            self.changes_made['files_analyzed'] = fix_results['files_processed']
            self.changes_made['imports_fixed'] = fix_results['imports_fixed']
        
        action = "would be" if dry_run else "were"
        self.logger.info(f"Import path fixing complete: {fix_results['imports_fixed']} imports {action} fixed in {fix_results['files_modified']} files")
        
        return fix_results
    
    def _fix_file_imports(self, file_path: Path, dry_run: bool = True) -> Dict[str, any]:
        """Fix imports in a specific file"""
        file_changes = {
            'file': str(file_path),
            'modified': False,
            'fixes_made': 0,
            'changes': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            modified_content = original_content
            
            # Apply import pattern fixes
            for old_pattern, new_pattern in self.import_patterns.items():
                matches = re.findall(old_pattern, modified_content)
                if matches:
                    modified_content = re.sub(old_pattern, new_pattern, modified_content)
                    file_changes['fixes_made'] += len(matches)
                    file_changes['changes'].append(f"Fixed {len(matches)} instances of pattern: {old_pattern}")
            
            # Check if file was modified
            if modified_content != original_content:
                file_changes['modified'] = True
                
                if not dry_run:
                    # Write the modified content back
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(modified_content)
                    
                    self.logger.debug(f"Fixed imports in: {file_path}")
        
        except Exception as e:
            self.logger.error(f"Error fixing imports in {file_path}: {str(e)}")
        
        return file_changes
    
    def normalize_module_paths(self, dry_run: bool = True) -> Dict[str, any]:
        """
        Normalize module paths to use consistent naming.
        
        Args:
            dry_run: If True, only analyze without making changes
            
        Returns:
            Results of the normalization process
        """
        self.logger.info(f"{'Analyzing' if dry_run else 'Normalizing'} module paths...")
        
        normalization_results = {
            'files_processed': 0,
            'paths_normalized': 0,
            'changes': [],
            'errors': []
        }
        
        # Common path normalizations
        path_normalizations = {
            # Ensure consistent module references
            r'from\s+src\.': 'from ',
            r'import\s+src\.': 'import ',
            r'from\s+auj_platform\.src\.': 'from ',
            r'import\s+auj_platform\.src\.': 'import ',
        }
        
        # Process files
        for file_path in self.src_path.rglob("*.py"):
            if self._should_analyze_file(file_path):
                normalization_results['files_processed'] += 1
                
                try:
                    file_normalized = self._normalize_file_paths(file_path, path_normalizations, dry_run)
                    if file_normalized['modified']:
                        normalization_results['paths_normalized'] += file_normalized['normalizations_made']
                        normalization_results['changes'].extend(file_normalized['changes'])
                
                except Exception as e:
                    error_msg = f"Error normalizing {file_path}: {str(e)}"
                    normalization_results['errors'].append(error_msg)
                    self.logger.error(error_msg)
        
        action = "would be" if dry_run else "were"
        self.logger.info(f"Path normalization complete: {normalization_results['paths_normalized']} paths {action} normalized")
        
        return normalization_results
    
    def _normalize_file_paths(self, file_path: Path, normalizations: Dict[str, str], dry_run: bool) -> Dict[str, any]:
        """Normalize paths in a specific file"""
        file_result = {
            'file': str(file_path),
            'modified': False,
            'normalizations_made': 0,
            'changes': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            modified_content = original_content
            
            # Apply normalizations
            for old_pattern, new_pattern in normalizations.items():
                matches = re.findall(old_pattern, modified_content)
                if matches:
                    modified_content = re.sub(old_pattern, new_pattern, modified_content)
                    file_result['normalizations_made'] += len(matches)
                    file_result['changes'].append(f"Normalized {len(matches)} instances of: {old_pattern}")
            
            # Save changes if modified
            if modified_content != original_content:
                file_result['modified'] = True
                
                if not dry_run:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(modified_content)
        
        except Exception as e:
            self.logger.error(f"Error normalizing {file_path}: {str(e)}")
        
        return file_result
    
    def run_minimal_cleanup(self, dry_run: bool = True) -> Dict[str, any]:
        """
        Run minimal path cleanup without disrupting existing architecture.
        
        Args:
            dry_run: If True, only analyze without making changes
            
        Returns:
            Complete cleanup results
        """
        self.logger.info("=== Starting minimal path cleanup ===")
        
        cleanup_results = {
            'analysis': {},
            'import_fixes': {},
            'path_normalization': {},
            'overall_status': 'unknown',
            'summary': {}
        }
        
        try:
            # 1. Analyze current state
            cleanup_results['analysis'] = self.analyze_import_structure()
            
            # 2. Fix import paths
            cleanup_results['import_fixes'] = self.fix_import_paths(dry_run)
            
            # 3. Normalize module paths
            cleanup_results['path_normalization'] = self.normalize_module_paths(dry_run)
            
            # Determine overall status
            total_issues = (
                len(cleanup_results['analysis']['import_issues']) +
                len(cleanup_results['analysis']['problematic_patterns'])
            )
            
            total_fixes = (
                cleanup_results['import_fixes']['imports_fixed'] +
                cleanup_results['path_normalization']['paths_normalized']
            )
            
            if total_issues == 0:
                cleanup_results['overall_status'] = 'clean'
            elif total_fixes > 0:
                cleanup_results['overall_status'] = 'improved'
            else:
                cleanup_results['overall_status'] = 'needs_attention'
            
            # Generate summary
            cleanup_results['summary'] = {
                'status': cleanup_results['overall_status'],
                'files_analyzed': cleanup_results['analysis']['python_files'],
                'import_issues_found': total_issues,
                'fixes_applied': total_fixes if not dry_run else 0,
                'potential_fixes': total_fixes if dry_run else 0,
                'files_would_be_modified': cleanup_results['import_fixes']['files_modified'],
                'dry_run': dry_run
            }
            
            # Log results
            if dry_run:
                self.logger.info(f"=== Cleanup analysis complete: {total_fixes} potential fixes identified ===")
            else:
                self.logger.info(f"=== Cleanup complete: {total_fixes} fixes applied ===")
        
        except Exception as e:
            cleanup_results['overall_status'] = 'error'
            cleanup_results['error'] = str(e)
            self.logger.error(f"Cleanup failed: {str(e)}")
        
        return cleanup_results
    
    def get_cleanup_summary(self) -> Dict[str, any]:
        """Get summary of all changes made during cleanup"""
        return {
            'changes_made': self.changes_made,
            'platform_root': str(self.platform_root),
            'src_path': str(self.src_path)
        }


def run_path_cleanup(platform_root: Optional[str] = None, dry_run: bool = True) -> Dict[str, any]:
    """
    Convenience function to run path cleanup.
    
    Args:
        platform_root: Optional platform root path
        dry_run: If True, only analyze without making changes
        
    Returns:
        Cleanup results
    """
    cleanup_utility = PathCleanupUtility(platform_root)
    return cleanup_utility.run_minimal_cleanup(dry_run)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AUJ Platform Path Cleanup')
    parser.add_argument('--platform-root', help='Platform root directory')
    parser.add_argument('--dry-run', action='store_true', help='Analyze only, do not make changes')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Run cleanup
    results = run_path_cleanup(args.platform_root, args.dry_run)
    
    # Print results
    print(f"\\nCleanup Status: {results['overall_status'].upper()}")
    print(f"Summary: {results['summary']}")
    
    if results['overall_status'] == 'error':
        print(f"Error: {results.get('error', 'Unknown error')}")
        exit(1)
    else:
        exit(0)