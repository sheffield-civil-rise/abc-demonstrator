#!/bin/sh

### This script accesses Hansel over SSH in order (1) to pull the latest code
### into Hansel's version of the repo, and (2) to run the demonstrator script
### on Hansel's hardware and software.

# Constants.
HANSEL_SSH_ID="hansel@172.16.67.13"
PATH_TO_REPO="G:\photogrammetry_e110a"
PATH_TO_ACTIVATE_SCRIPT="C:\Users\hansel\Anaconda3\Scripts\activate"
ENV_NAME="demonstrator"

# Exit on first error.
set -e

# Extract arguments.
next_is_git_url=false
git_url=false
for argument in $@; do
    if [ $argument = "--git-url" ]; then
        next_is_git_url=true
    elif $next_is_git_url; then
        git_url=$argument
        next_is_git_url=false
    fi
done
if [ ! git_url ]; then
    echo "You need to provide an argument giving the Git URL."
    echo "Place this after the --git-url flag."
    exit 1
fi

# Let's get cracking.
ssh $HANSEL_SSH_ID <<ENDSSH
    git -C $PATH_TO_REPO checkout restructuring0
    git -C $PATH_TO_REPO pull $git_url restructuring0
    IF %ERRORLEVEL% NEQ 0 ( 
        exit 1
    )
    $PATH_TO_ACTIVATE_SCRIPT $ENV_NAME
    python $PATH_TO_REPO\run_demonstrator.py
ENDSSH
