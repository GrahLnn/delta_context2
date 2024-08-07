from itertools import chain

def flatten(list_of_lists):
    return list(chain.from_iterable(list_of_lists))

def drop_duplicate(lst):
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]