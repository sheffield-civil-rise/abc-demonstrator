"""
This code defines a function which generates a checksum string for a given file.
"""

# Standard imports.
import hashlib
import sys

# Local constants.
CHUNK_SIZE = 4096

#############
# FUNCTIONS #
#############

def make_checksum(path_to_file, chunk_size=CHUNK_SIZE):
    """ The function in question. """
    hash_md5 = hashlib.md5()
    with open(path_to_file, "rb") as hash_me:
        for chunk in iter(lambda: hash_me.read(chunk_size), b""):
            hash_md5.update(chunk)
    result = hash_md5.hexdigest()
    return result

###################
# RUN AND WRAP UP #
###################

def run():
    """ Run this file. """
    if len(sys.argv) > 1:
        print(make_checksum(sys.argv[1]))

if __name__ == "__main__":
    run()
