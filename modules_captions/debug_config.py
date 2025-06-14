#!/usr/bin/env python3
"""
Debug script to investigate configuration duplication issue.
"""

import json
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_captions import load_provider_config, CaptionExtractor

def debug_config():
    """Debug configuration loading and status generation."""
    print("=== DEBUGGING CONFIGURATION ===")
    
    # Test load_provider_config
    print("\n1. Testing load_provider_config('gemini'):")
    config = load_provider_config('gemini')
    print("Config keys:", list(config.keys()))
    print("Client config:", config.get('client_config', {}))
    
    # Add missing required keys
    config['db_path'] = 'captions.db'
    config['root_dir'] = 'test_dir'
    
    # Test CaptionExtractor initialization
    print("\n2. Testing CaptionExtractor initialization:")
    try:
        extractor = CaptionExtractor(config)
        print("Extractor config keys:", list(extractor.config.keys()))
        print("Extractor client config:", extractor.config.get('client_config', {}))
        
        # Test get_status
        print("\n3. Testing get_status:")
        status = extractor.get_status()
        print("Status keys:", list(status.keys()))
        
        # Check if config in status has duplicates
        status_config = status.get('config', {})
        print("Status config keys:", list(status_config.keys()))
        print("Status client config:", status_config.get('client_config', {}))
        
        # Print the actual JSON to see duplication
        print("\n4. Detailed analysis:")
        print("Type of status:", type(status))
        print("Status config type:", type(status_config))
        
        # Check if client_config appears multiple times in the dict
        client_config = status_config.get('client_config', {})
        print("Client config type:", type(client_config))
        print("Client config model:", client_config.get('model', 'NOT_FOUND'))
        
        # Check raw string representation
        print("\n5. Raw string representation:")
        print(repr(status_config))
        
        print("\n6. JSON output:")
        json_output = json.dumps(status, indent=2, ensure_ascii=False)
        print(json_output)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_config()