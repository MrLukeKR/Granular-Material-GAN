from Settings import MessageTools as mt
from Settings.MessageTools import print_notice


def calculate_all(core):
    results = list()

    results.append(calculate_composition(core))
    results.append(calculate_tortuosity(core))
    results.append(calculate_euler_number(core))
    results.append(calculate_average_void_diameter(core))

    return results


def calculate_composition(core):
    print_notice("Calculating Core Compositions...", mt.MessagePrefix.INFORMATION)
    raise NotImplementedError


def calculate_tortuosity(core):
    print_notice("Calculating Core Tortuosity...", mt.MessagePrefix.INFORMATION)
    raise NotImplementedError


def calculate_euler_number(core):
    print_notice("Calculating Core Euler Number...", mt.MessagePrefix.INFORMATION)
    raise NotImplementedError


def calculate_average_void_diameter(core):
    print_notice("Calculating Core Average Void Diameter...", mt.MessagePrefix.INFORMATION)
    raise NotImplementedError
