#!/bin/sh

### This code quickly pushes any changes and runs the new code on Hansel.

# Constants.
BRANCH="restructuring0"

# Fail on the first error.
set -e

# Update flags.
old_flag=false
for argument in $@; do
    if [ $argument = "--old" ]; then
        old_flag=true
    fi
done

# Let's get cracking...
git add .
git commit -m "Debugging..." || echo "Let's not worry about that..."
git push origin $BRANCH
if $old_flag; then
    python3 check_with_hansel.py --old
else
    python3 check_with_hansel.py
fi
