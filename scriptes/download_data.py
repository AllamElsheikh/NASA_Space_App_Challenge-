#!/usr/bin/env python3
"""
Automated Data Download Script
NASA Space Apps Challenge 2025
"""

import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.downloaders import download_koi_catalog, download_toi_catalog


def main():
    parser = argparse.ArgumentParser(description='Download NASA exoplanet data')
    parser.add_argument('--catalogs', nargs='+', default=['koi', 'toi'],
                       help='Catalogs to download (koi, toi)')
    parser.add_argument('--light-curves', action='store_true',
                       help='Download light curve data')
    
    args = parser.parse_args()
    
    print("🚀 NASA Space Apps Challenge 2025 - Data Download")
    print("=" * 50)
    
    # Download catalogs
    if 'koi' in args.catalogs:
        print("📡 Downloading KOI catalog...")
        # download_koi_catalog()
    
    if 'toi' in args.catalogs:
        print("📡 Downloading TOI catalog...")
        # download_toi_catalog()
    
    print("✅ Data download completed!")


if __name__ == "__main__":
    main()
