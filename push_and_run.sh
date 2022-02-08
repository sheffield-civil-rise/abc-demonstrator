#!/bin/sh

### This code quickly pushes any changes and runs the new code on Hansel.

# Constants.
BRANCH="restructuring0"

# Fail on the first error.
set -e

# Parse arguments.
next_is_git_message=false
git_message="Debugging..."
for argument in $@; do
    if [ $argument = "--message" ] || [ $argument = "-m" ]; then
        next_is_git_message=true
    elif $next_is_git_message; then
        git_message=$argument
        next_is_git_message=false
    fi
done

# Let's get cracking...
git add .
git commit -m "$git_message"
git push origin $BRANCH
python3 check_with_hansel.py
