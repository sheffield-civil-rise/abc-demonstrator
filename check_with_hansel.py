"""
This code defines a script which checks with Hansel whether the demonstrator
is running correctly.
"""

# Standard imports.
import json
import os
import subprocess
from pathlib import Path

# Constants.
PATH_TO_HOME = str(Path.home())
DEFAULT_PATH_TO_REPO = os.path.join(PATH_TO_HOME, "photogrammetry_e110a")
DEFAULT_PATH_TO_SCRIPT = os.path.join(DEFAULT_PATH_TO_REPO, "run_on_hansel.sh")
DEFAULT_PATH_TO_REPO_ON_HANSEL = r"G:\photogrammetry_e110a"
DEFAULT_PATH_TO_ACTIVATE_SCRIPT = r"C:\Users\hansel\Anaconda3\Scripts\activate"
DEFAULT_ENV_NAME = "demonstrator"
DEFAULT_REPO_URL = "github.com/tomhosker/photogrammetry_e110a.git"
DEFAULT_PATH_TO_SECURITY_FILE = \
    os.path.join(PATH_TO_HOME, "hansel_security.json")
DEFAULT_ENCODING = "utf-8"
DEFAULT_BRANCH = "restructuring0"

#############
# FUNCTIONS #
#############

def get_security_dict(
        path_to=DEFAULT_PATH_TO_SECURITY_FILE,
        encoding=DEFAULT_ENCODING
    ):
    """ Read in the JSON file as a dictionary """
    with open(path_to, "r", encoding=encoding) as token_file:
        json_str = token_file.read()
    result = json.loads(json_str)
    return result

def run_on_hansel(
        personal_access_token,
        ssh_id,
        ssh_password,
        path_to_repo=DEFAULT_PATH_TO_REPO_ON_HANSEL,
        path_to_activate_script=DEFAULT_PATH_TO_ACTIVATE_SCRIPT,
        env_name=DEFAULT_ENV_NAME,
        repo_url=DEFAULT_REPO_URL,
        path_to_script=DEFAULT_PATH_TO_SCRIPT,
        branch=DEFAULT_BRANCH,
        hide_output=False
    ):
    """ Call the shell script with the correct arguments. """
    git_url = "https://"+personal_access_token+"@"+repo_url
    arguments = [
        "sh", path_to_script,
        "--ssh-id", ssh_id,
        "--ssh-password", ssh_password,
        "--git-url", git_url,
        "--branch", branch,
        "--path-to-repo", path_to_repo,
        "--path-to-activate-script", path_to_activate_script,
        "--env-name", env_name
    ]
    try:
        if hide_output:
            subprocess.run(arguments, check=True, stdout=subprocess.DEVNULL)
        else:
            subprocess.run(arguments, check=True)
    except subprocess.CalledProcessError:
        return False
    return True

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    security_dict = get_security_dict()
    result = \
        run_on_hansel(
            security_dict["personal_access_token"],
            security_dict["ssh_id"],
            security_dict["ssh_password"]
        )
    return result

if __name__ == "__main__":
    run()
