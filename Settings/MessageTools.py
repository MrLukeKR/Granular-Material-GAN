import os

from colorama import Fore, Style, init

init_complete = False
using_pycharm = False if 'PYCHARM_HOSTED' in os.environ else None


class MessagePrefix:
    INFORMATION = "[" + Fore.CYAN + "INFORMATION" + Fore.RESET + "]",
    WARNING = "[" + Fore.YELLOW + "WARNING" + Fore.RESET + "]",
    ERROR = "[" + Fore.RED + "ERROR" + Fore.RESET + "]",
    SUCCESS = "[" + Fore.GREEN + "SUCCESS" + Fore.RESET + "]",


def print_notice(message, message_type=MessagePrefix.INFORMATION, end="\r\n"):
    global init_complete, using_pycharm

    if not init_complete:
        init(convert=using_pycharm, strip=using_pycharm)
        init_complete = True

    print(message_type[0] + " " + message, end=end)
