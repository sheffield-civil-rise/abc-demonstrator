"""
This code defines a script which runs the demonstrator script on Hansel.
"""

# Standard imports.
import json
import os
import subprocess
import sys
from pathlib import Path

# Constants.
PATH_TO_HOME = str(Path.home())
DEFAULT_PATH_TO_REPO = os.path.join(PATH_TO_HOME, "photogrammetry_e110a")
DEFAULT_PATH_TO_SCRIPT = \
    os.path.join(DEFAULT_PATH_TO_REPO, "run_on_hansel.sh")
DEFAULT_PATH_TO_REPO_ON_HANSEL = r"G:\photogrammetry_e110a"
DEFAULT_PATH_TO_ACTIVATE_SCRIPT = r"C:\Users\hansel\Anaconda3\Scripts\activate"
DEFAULT_ENV_NAME = "demonstrator"
DEFAULT_REPO_URL = "github.com/tomhosker/photogrammetry_e110a.git"
DEFAULT_PATH_TO_SECURITY_FILE = \
    os.path.join(PATH_TO_HOME, "hansel_security.json")
DEFAULT_ENCODING = "utf-8"

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

def get_current_branch():
    """ Get the current branch checked out in THIS copy of the repo. """
    out = \
        subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True
        )
    result = out.stdout
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
        branch=get_current_branch(),
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

def run_on_hansel_with_auth(
        path_to_repo=DEFAULT_PATH_TO_REPO_ON_HANSEL,
        path_to_activate_script=DEFAULT_PATH_TO_ACTIVATE_SCRIPT,
        env_name=DEFAULT_ENV_NAME,
        repo_url=DEFAULT_REPO_URL,
        path_to_script=DEFAULT_PATH_TO_SCRIPT,
        branch=get_current_branch(),
        hide_output=False
    ):
    """ Generate the required auth data, and then call the script. """
    security_dict = get_security_dict()
    result = \
        run_on_hansel(
            security_dict["personal_access_token"],
            security_dict["ssh_id"],
            security_dict["ssh_password"],
            path_to_repo=path_to_repo,
            path_to_activate_script=path_to_activate_script,
            env_name=env_name,
            repo_url=repo_url,
            path_to_script=path_to_script,
            branch=branch,
            hide_output=hide_output
        )
    return result

def print_encased(message, symbol="#"):
    """ Print the message encased in hashes. """
    message_line = symbol+" "+message+" "+symbol
    hashes = ""
    for _ in range(len(message_line)):
        hashes = hashes+symbol
    print(" ")
    print(hashes)
    print(message_line)
    print(hashes)
    print(" ")

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    if run_on_hansel_with_auth():
        print_encased("The process which you ran on Hansel succeeded.")
        sys.exit(0)
    print_encased("Sorry, but the process which you ran on Hansel failed.")
    sys.exit(1)

if __name__ == "__main__":
    run()
