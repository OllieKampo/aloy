from dataclasses import make_dataclass
from functools import cache
import functools
import typing
from weakref import WeakValueDictionary
import collections.abc

from auxiliary.hashing import hash_all

def create_if_not_exists_in_slots(class_dict: dict[str, typing.Any], **class_attr_names: dict[str, str]) -> None:
    """Create class attributes if they do not exist in the `__slots__` of a class dictionary."""
    class_dict_copy = class_dict.copy()
    
    if (slots := class_dict.get("__slots__")) is not None:
        missing: dict[str, str] = {attr : desc
                                   for attr, desc
                                   in class_attr_names.items()
                                   if attr not in slots}
        
        if missing:
            if isinstance(slots, str):
                slots = (slots, *missing)
            elif isinstance(slots, (list, tuple, set)):
                slots = type(slots)((*slots, *missing))
            elif isinstance(slots, (dict, collections.abc.Mapping)):
                slots = type(slots)(slots | missing)
            else:
                try:
                    iter(slots)
                    slots = type(slots)((*slots, *missing))
                except TypeError as e:
                    raise TypeError("Cannot determine type of __slots__ attribute.") from e
            class_dict_copy["__slots__"] = slots
        
    return class_dict_copy

class CachedInstancesClass(type):
    """
    Instances are immutable and hashable.
    """
    
    __cache__ = WeakValueDictionary()
    
    def __new__(cls, cls_name: str, bases: tuple[str], class_dict: dict) -> type:
        
        
        original_new = class_dict["__new__"]
        @functools.wraps(original_new)
        def __new__(*args, **kwargs):
            hash_: int = hash_all(*args, **kwargs)
            if (cached_instance := cls.__cache__.get(hash_)) is not None:
                return cached_instance
            instance = original_new(*args, **kwargs)
            cls.__cache__[hash_] = instance
            return instance
        
        return super().__new__(cls, cls_name, bases, class_dict)