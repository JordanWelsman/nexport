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