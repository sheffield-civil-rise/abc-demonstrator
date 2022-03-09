"""
This code defines a script which allows the user to access Hansel's SSH command
line.
"""

# Standard imports.
import subprocess

# Local imports.
from run_on_hansel import (
    DEFAULT_PATH_TO_ACTIVATE_SCRIPT, DEFAULT_ENV_NAME, get_security_dict
)

#############
# FUNCTIONS #
#############

def access_hansel_ssh(
        path_to_activate_script=DEFAULT_PATH_TO_ACTIVATE_SCRIPT,
        env_name=DEFAULT_ENV_NAME
    ):
    """ Access Hansel's SSH command line. """
    security_dict = get_security_dict()
    ssh_password = security_dict["ssh_password"]
    ssh_id = security_dict["ssh_id"]
    arguments = ["sshpass", "-p"+ssh_password, "ssh", ssh_id]
    print(
        "Accessing Hansel's SSH command line...\n\n"+
        "Run:\n\n"+
        "    "+path_to_activate_script+" "+env_name+"\n\n"+
        "to enter the demonstrator environment.\n\n"+
        "Type `exit` to leave."
    )
    subprocess.run(arguments, check=True)

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    access_hansel_ssh()

if __name__ == "__main__":
    run()
