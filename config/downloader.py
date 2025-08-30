"""
NASA Data Downloaders
NASA Space Apps Challenge 2025
"""

import requests
import pandas as pd
from astroquery.mast import Catalogs


def download_koi_catalog():
    """Download Kepler Objects of Interest catalog."""
    pass


def download_toi_catalog():
    """Download TESS Objects of Interest catalog."""
    pass


def download_light_curves(target_ids, mission="Kepler"):
    """Download light curve data for given targets."""
    pass


def fetch_from_mast_api(endpoint, params):
    """Generic function to fetch data from MAST API."""
    pass
