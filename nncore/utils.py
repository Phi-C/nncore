import threading
from typing import Tuple, Any, Dict


class SingletonMeta(type):
    """
    This is a metaclass for creating a Singleton class.

    Example:
        .. code-block:: python

        class MySingletonClass(metaclass=SingletonMeta):
            pass
    """

    _singleton_lock = threading.Lock()

    def __new__(mcls, name: str, bases: Tuple[type], ns: Dict[str, Any], **kw) -> Any:
        """
        Create a new class type, not instance of this class
        """
        return super().__new__(mcls, name, bases, ns)

    def __init__(cls, name: str, bases: Tuple[type], ns: Dict[str, Any], **kw) -> Any:
        """
        Initialize the class itself
        """
        cls._instance_cache: Dict[Tuple, Any] = {}
        super().__init__(name, bases, ns)

    def __call__(cls, *args, **kwargs):
        """
        Main logic for singleton class, it control the process of class instantiation.
        """
        cache_key = (args, frozenset(kwargs.items()) if kwargs else None)
        with cls._singleton_lock:
            instance = cls._instance_cache.get(cache_key)
            if not instance:
                instance = super().__call__(*args, **kwargs)
                cls._instance_cache[cache_key] = instance
        return instance
