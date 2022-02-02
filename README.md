# Photogrammetry E110A

This repository holds the code which I worked on for a photogrammetry project in room E110A at the University of Sheffield.

## Installation

The code in this repo calls a script which - through the power of SSH - runs on a high-spec machine called Hansel. Therefore, the installation required to run the code on this machine is actually relatively light. It consists of the following steps:

1. Install the University of Sheffield VPN.
1. Clone this repo into your home directory.
1. If you're working with a device which uses **APT** as the package manager, run `python3 install.py`. Otherwise, install PIP3, and install the packages in pip_requirements.txt manually.
1. Obtain or create a hansel_security.json file, and place it in your home directory. (See below for a template of what this file should look like.)

### Example hansel_security.json

```json
{
    "personal_access_token": "PERSONAL-ACCESS-TOKEN-FOR-THIS-REPO",
    "ssh_password": "HANSEL-SSH-PASSWORD"
}
```

## Usage

* If you're working from a non-university machine, it may be necessary to start the University of Sheffield VPN in order to access Hansel.
* Run the code in this repo by calling `python3 check_with_hansel.py`, and providing Hansel's SSH password when prompted.
* You can also run the same code manually on Hansel itself by calling `run_demonstrator.py`.
