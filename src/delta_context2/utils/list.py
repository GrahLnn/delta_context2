from itertools import chain

def flatten(list_of_lists):
    return list(chain.from_iterable(list_of_lists))