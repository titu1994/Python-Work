import inspect


def custom_dir(c, add):
    return dir(type(c)) + list(c.__dict__.keys()) + add


class BaseComposite:
    "Base class for attr accesses in `self.__extra_args__` passed down to `self.default`"

    @property
    def __extra_args__(self):
        if not hasattr(self, 'default'):
            self.default = []

        return [o for o in dir(self.default)
                if not o.startswith('_')]

    def __getattr__(self, k):
        if k in self.__extra_args__:
            return getattr(self.default, k)

        raise AttributeError(k)

    def __dir__(self):
        return custom_dir(self, self.__extra_args__)
