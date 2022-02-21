#!/bin/sh

### This script accesses Hansel over SSH in order (1) to pull the latest code
### into Hansel's version of the repo, and (2) to run the repo's automated tests
### on Hansel's hardware and software.

# Constants.
PATH_TO_REPO="G:\photogrammetry_e110a"
PATH_TO_ACTIVATE_SCRIPT="C:\Users\hansel\Anaconda3\Scripts\activate"
ENV_NAME="demonstrator"
PATH_TO_DEMONSTRATOR_SCRIPT="$PATH_TO_REPO\run_tests.py"

# Exit on first error.
set -e

# Extract arguments.
next_is_ssh_id=false
ssh_id=false
next_is_ssh_password=false
ssh_password=false
next_is_git_url=false
git_url=false
next_is_branch=false
branch="master"
next_is_path_to_repo=false
path_to_repo=false
next_is_path_to_activate_script=false
path_to_activate_script=false
next_is_env_name=false
env_name=false
for argument in $@; do
    if [ $argument = "--git-url" ]; then
        next_is_git_url=true
    elif $next_is_git_url; then
        git_url=$argument
        next_is_git_url=false
    elif [ $argument = "--branch" ]; then
        next_is_branch=true
    elif $next_is_branch; then
        branch=$argument
        next_is_branch=false
    elif [ $argument = "--ssh-id" ]; then
        next_is_ssh_id=true
    elif $next_is_ssh_id; then
        ssh_id=$argument
        next_is_ssh_id=false
    elif [ $argument = "--ssh-password" ]; then
        next_is_ssh_password=true
    elif $next_is_ssh_password; then
        ssh_password=$argument
        next_is_ssh_password=false
    elif [ $argument = "--path-to-repo" ]; then
        next_is_path_to_repo=true
    elif $next_is_path_to_repo; then
        path_to_repo=$argument
        next_is_path_to_repo=false
    elif [ $argument = "--path-to-activate-script" ]; then
        next_is_path_to_activate_script=true
    elif $next_is_path_to_activate_script; then
        path_to_activate_script=$argument
        next_is_path_to_activate_script=false
    elif [ $argument = "--env-name" ]; then
        next_is_env_name=true
    elif $next_is_env_name; then
        env_name=$argument
        next_is_env_name=false
    fi
done
if [ ! git_url ]; then
    echo "You need to provide an argument giving the Git URL."
    echo "Place this after the --git-url flag."
    exit 1
fi

# Let's get cracking.
sshpass -p$ssh_password ssh $ssh_id <<ENDSSH
    git -C $PATH_TO_REPO checkout $branch
    git -C $PATH_TO_REPO pull $git_url $branch
    IF %ERRORLEVEL% NEQ 0 (
        exit 1
    )
    $PATH_TO_ACTIVATE_SCRIPT $ENV_NAME
    cd $PATH_TO_REPO
    python $PATH_TO_DEMONSTRATOR_SCRIPT
ENDSSH
