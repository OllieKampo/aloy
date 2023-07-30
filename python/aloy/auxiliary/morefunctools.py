
import functools


def call_on(name: str, iterable: bool = False):
    def inner(func):
        if iterable:
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                result = self.func(*args, **kwargs)
                others = (
                    item(*args, **kwargs)
                    for item in getattr(self, name)
                )
                return result, others
        else:
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                result = self.func(*args, **kwargs)
                other = getattr(self, name)(*args, **kwargs)
                return result, other
        return wrapper
    return inner
