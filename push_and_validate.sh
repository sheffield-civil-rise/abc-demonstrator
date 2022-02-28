#!/bin/sh

### This code quickly pushes any changes and runs the new code on Hansel.

# Constants.
BRANCH="restructuring1"

# Fail on the first error.
set -e

# Let's get cracking...
git add .
git commit -m "Debugging..." || echo "Let's not worry about that..."
git push origin $BRANCH
python3 continuous_integration.py