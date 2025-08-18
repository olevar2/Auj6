#!/usr/bin/env python3
"""
Setup script for AUJ Platform
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
req_file = this_directory / "auj_platform" / "requirements.txt"
if req_file.exists():
    with open(req_file, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="auj-platform",
    version="1.0.0",
    description="Advanced AI Trading Platform for Humanitarian Impact",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AUJ Platform Development Team",
    author_email="support@auj-platform.com",
    url="https://github.com/olevar2/AUJ",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-asyncio>=0.21.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
        ],
        'docs': [
            'sphinx>=7.0.0',
            'sphinx-rtd-theme>=1.3.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'auj-platform=auj_platform.src.main:main',
            'auj-dashboard=auj_platform.dashboard.launch_dashboard:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    keywords="trading, ai, machine-learning, finance, humanitarian",
    project_urls={
        "Bug Reports": "https://github.com/olevar2/AUJ/issues",
        "Source": "https://github.com/olevar2/AUJ",
        "Documentation": "https://github.com/olevar2/AUJ/docs",
    },
)
