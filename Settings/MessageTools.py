import os

from colorama import Fore, init

init_complete = False
using_pycharm = False if 'PYCHARM_HOSTED' in os.environ else None


class MessagePrefix:
    INFORMATION = "[" + Fore.CYAN + "INFORMATION" + Fore.RESET + "]",
    WARNING = "[" + Fore.YELLOW + "WARNING" + Fore.RESET + "]",
    ERROR = "[" + Fore.RED + "ERROR" + Fore.RESET + "]",
    SUCCESS = "[" + Fore.GREEN + "SUCCESS" + Fore.RESET + "]",
    DEBUG = "[" + Fore.MAGENTA + "DEBUG" + Fore.RESET + "]",


def get_notice(message, message_type=MessagePrefix.INFORMATION):
    return message_type[0] + " " + message


def print_notice(message, message_type=MessagePrefix.INFORMATION, end="\r\n", flush=True):
    global init_complete, using_pycharm

    if not init_complete:
        init(convert=using_pycharm, strip=using_pycharm)
        init_complete = True

    print(message_type[0] + " " + message, end=end, flush=flush)
