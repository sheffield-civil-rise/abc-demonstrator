"""
This code defines a minimal continuous integration script.
"""

# Standard imports.
import argparse
import os
import subprocess
from glob import glob

# Local imports.
from run_on_hansel import run_on_hansel_with_auth, DEFAULT_PATH_TO_REPO
from validate import run_tests as run_tests_locally

# Local constants.
PATH_TO_SCRIPT = os.path.join(DEFAULT_PATH_TO_REPO, "validate_on_hansel.sh")
DEFAULT_LINTER_CONFIGS = {
    "max_line_length": 80,
    "messages_to_disable": ("import-error", "too-many-instance-attributes"),
    "min_score": 9,
    "patterns_to_ignore": ("**/test_*.py",)
}

#############
# FUNCTIONS #
#############

def make_parser():
    """ Return a parser argument. """
    result = \
        argparse.ArgumentParser(description="A continuous integration script")
    result.add_argument(
        "--no-lint",
        action="store_true",
        default=False,
        dest="no_lint",
        help="Do not run the linter"
    )
    result.add_argument(
        "--no-test",
        action="store_true",
        default=False,
        dest="no_test",
        help="Do not run the tests"
    )
    result.add_argument(
        "--stop-on-failure",
        action="store_true",
        default=False,
        dest="stop_on_failure",
        help="Stop script on the first failure"
    )
    result.add_argument(
        "--test-locally",
        action="store_true",
        default=False,
        dest="test_locally",
        help="Run the tests locally, i.e. not via SSH"
    )
    return result

def make_files_to_ignore(patterns_to_ignore):
    """ Make a string list of files from a list of patterns. """
    file_set = set()
    for pattern in patterns_to_ignore:
        file_list = glob(pattern)
        for file_path in file_list:
            file_set.add(os.path.basename(file_path))
    result = ",".join(list(file_set))
    return result

def run_linter(configs=None):
    """ Run PyLint on this repo. """
    if configs is None:
        configs = DEFAULT_LINTER_CONFIGS
    result = True
    source_file_paths = glob("**/*.py")
    arguments = [
        "pylint",
        "--fail-under="+str(configs["min_score"]),
        "--disable="+(",".join(configs["messages_to_disable"])),
        "--ignore="+make_files_to_ignore(configs["patterns_to_ignore"]),
        "--max-line-length="+str(configs["max_line_length"])
    ]
    arguments = arguments+source_file_paths
    try:
        subprocess.run(arguments, check=True)
    except subprocess.CalledProcessError:
        result = False
    return result

def run_continuous_integration(
        lint=True, test=True, stop_on_failure=False, test_locally=False
    ):
    """ Execute a minimal continuous integration routine. """
    lint_result, test_result = True, True
    if lint:
        lint_result = run_linter()
        if (not lint_result) and stop_on_failure:
            return False
    if test:
        if test_locally:
            test_result = run_tests_locally()
        else:
            test_result = run_on_hansel_with_auth(path_to_script=PATH_TO_SCRIPT)
        if (not test_result) and stop_on_failure:
            return False
    if lint_result and test_result:
        return True
    return False

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    parser = make_parser()
    arguments = parser.parse_args()
    result = \
        run_continuous_integration(
            lint=(not arguments.no_lint),
            test=(not arguments.no_test),
            stop_on_failure=arguments.stop_on_failure,
            test_locally=arguments.test_locally
        )
    return result

if __name__ == "__main__":
    run()
