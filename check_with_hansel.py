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
DEFAULT_REPO_URL = "github.com/tomhosker/photogrammetry_e110a.git"
DEFAULT_PATH_TO_SECURITY_FILE = \
    os.path.join(PATH_TO_HOME, "hansel_security.json")
DEFAULT_ENCODING = "utf-8"
DEFAULT_BRANCH = "restructuring0"

#############
# FUNCTIONS #
#############

def get_personal_access_token(
        path_to=DEFAULT_PATH_TO_SECURITY_FILE,
        encoding=DEFAULT_ENCODING
    ):
    """ Read in the string from a file. """
    with open(path_to, "r", encoding=encoding) as token_file:
        json_str = token_file.read()
    json_dict = json.loads(json_str)
    result = json_dict["personal_access_token"]
    return result

def run_on_hansel(
        personal_access_token,
        repo_url=DEFAULT_REPO_URL,
        path_to_script=DEFAULT_PATH_TO_SCRIPT,
        branch=DEFAULT_BRANCH,
        hide_output=False
    ):
    """ Call the shell script with the correct arguments. """
    git_url = "https://"+personal_access_token+"@"+repo_url
    arguments = ["sh", path_to_script, "--git-url", git_url, "--branch", branch]
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
    token = get_personal_access_token()
    return run_on_hansel(token)

if __name__ == "__main__":
    run()
