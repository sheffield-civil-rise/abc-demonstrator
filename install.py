"""
This code defines the install script for this repo.
"""

# Standard imports.
import subprocess

# Local constants.
PIP_APT_PACKAGE_NAME = "python3-pip"
PIP_COMMAND = "pip3"
DEFAULT_PATH_TO_PIP_REQUIREMENTS = "pip_requirements.txt"
REQUIRED_APT_PACKAGES = ["sshpass"]

#############
# FUNCTIONS #
#############

def install_pip_and_packages(
        path_to_pip_requirements=DEFAULT_PATH_TO_PIP_REQUIREMENTS
    ):
    """ Run the install script. """
    subprocess.run(
        ["sudo", "apt", "install", PIP_APT_PACKAGE_NAME],
        check=True
    )
    subprocess.run(
        [PIP_COMMAND, "install", "-r", path_to_pip_requirements],
        check=True
    )

def install_apt_packages():
    """ Install the required APT packages. """
    for package in REQUIRED_APT_PACKAGES:
        subprocess.run(["sudo", "apt", "install", package], check=True)

def install(path_to_pip_requirements=DEFAULT_PATH_TO_PIP_REQUIREMENTS):
    """ Run the install script. """
    install_pip_and_packages(path_to_pip_requirements=path_to_pip_requirements)
    install_apt_packages()

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    install()

if __name__ == "__main__":
    run()
