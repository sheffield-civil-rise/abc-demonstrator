"""
This code modifies how PyTest runs.
"""

# Local constants.
ORDERED_TEST_MODULES = (
    "tests.test_local_configs.py",
    "tests.test_demonstrator.py"
)

######################
# RESERVED FUNCTIONS #
######################

def pytest_collection_modifyitems(items):
    """ Modifies test items in place to ensure test modules run in a given
    order. """
    module_mapping = {item: item.module.__name__ for item in items}
    sorted_items = items.copy()
    # Iteratively move tests of each module to the end of the test queue.
    for module in ORDERED_TEST_MODULES:
        sorted_items = (
            [ item for item in sorted_items if module_mapping[item] != module ]+
            [ item for item in sorted_items if module_mapping[item] == module ]
        )
    items[:] = sorted_items
