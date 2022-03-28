"""
This code defines a script which in turn runs the demonstrator script.
"""

# Standard imports.
import sys
import traceback
from pathlib import Path

# Local imports.
sys.path.append(str(Path(__file__).parent/"source"))
from demonstrator import Demonstrator

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    debug = False
    if "--debug" in sys.argv:
        debug = True
    demonstrator = Demonstrator(debug=debug)
    try:
        demonstrator.demonstrate()
    except Exception:
        traceback.print_exc()
        print("Sorry, but the demonstrator failed.")
        return False
    print("Demonstrator ran successfully!")
    return True

if __name__ == "__main__":
    run()
