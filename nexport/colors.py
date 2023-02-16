# COPYRIGHT NOTICE

# “Neural Network Export Package (nexport) v0.4.6” Copyright (c) 2023,
# The Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.

# If you have questions about your rights to use or distribute this software,
# please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

# NOTICE. This Software was developed under funding from the U.S. Department
# of Energy and the U.S. Government consequently retains certain rights. As
# such, the U.S. Government has been granted for itself and others acting on
# its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
# Software to reproduce, distribute copies to the public, prepare derivative
# works, and perform publicly and display publicly, and to permit others to do so.


class Color:
    DEFAULT =       '\033[39m'
    BLACK =         '\033[30m'
    RED =           '\033[31m'
    GREEN =         '\033[32m'
    YELLOW =        '\033[33m'
    BLUE =          '\033[34m'
    MAGENTA =       '\033[35m'
    CYAN =          '\033[36m'
    LIGHTGRAY =     '\033[37m'
    DARKGRAY =      '\033[90m'
    LIGHTRED =      '\033[91m'
    LIGHTGREEN =    '\033[92m'
    LIGHTYELLOW =   '\033[93m'
    LIGHTBLUE =     '\033[94m'
    LIGHTMAGENTA =  '\033[95m'
    LIGHTCYAN =     '\033[96m'
    WHITE =         '\033[97m'

    def test():
        print(f"{Color.DEFAULT}█{Color.BLACK}█{Color.DARKGRAY}█{Color.LIGHTGRAY}█{Color.WHITE}█")
        print(f"{Color.RED}█{Color.GREEN}█{Color.YELLOW}█{Color.BLUE}█{Color.MAGENTA}█{Color.CYAN}█")
        print(f"{Color.LIGHTRED}█{Color.LIGHTGREEN}█{Color.LIGHTYELLOW}█{Color.LIGHTBLUE}█{Color.LIGHTMAGENTA}█{Color.LIGHTCYAN}█{Color.DEFAULT}")