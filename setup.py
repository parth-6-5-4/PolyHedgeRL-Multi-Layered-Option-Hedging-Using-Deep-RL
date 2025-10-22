"""
Setup configuration for PolyHedgeRL package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'readme.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name='polyhedge-rl',
    version='0.1.0',
    author='Parth Dambhare',
    description='Multi-Layered Option Hedging Using Deep Reinforcement Learning',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/parth-6-5-4/PolyHedgeRL-Multi-Layered-Option-Hedging-Using-Deep-RL',
    packages=find_packages(exclude=['tests', 'notebooks', 'scripts']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Office/Business :: Financial :: Investment',
    ],
    python_requires='>=3.9',
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
        ],
        'docs': [
            'sphinx>=4.5.0',
            'sphinx-rtd-theme>=1.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'polyhedge-train=scripts.train_agents:main',
            'polyhedge-evaluate=scripts.evaluate_performance:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
