"""Installation script for the 'dropbear_rl_lab' python package."""

import os
import toml

from setuptools import setup, find_packages

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "psutil",
    "moviepy",  # For video recording functionality
]

# Optional dependencies for remote A100 workers (no IsaacLab)
EXTRAS_REQUIRE = {
    "remote": [
        "gymnasium>=0.29",
        "numpy>=1.24",
        "torch>=2.0",
        # Note: rsl_rl should be installed separately via pip
        # as it's not a standard PyPI package
    ],
}

# Installation operation
setup(
    name="dropbear_rl_lab",
    packages=find_packages(),
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    license="Apache 2.0",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.5.0",
    ],
    zip_safe=False,
)
