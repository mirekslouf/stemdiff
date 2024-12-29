# __main__.py = top-level code environment
# Usage: (1) __main__ in programs (2) __main__.py in packages
# More information: https://docs.python.org/3/library/__main__.html
# Here: To print the the acknowledgement when we run: python -m stemdiff
# (the acknowledgement function is defined in the sister __init__.py file

from stemdiff import acknowledgement

if __name__ == '__main__': acknowledgement()
