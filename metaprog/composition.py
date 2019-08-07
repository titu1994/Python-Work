

def custom_dir(c, add):
    return dir(type(c)) + list(c.__dict__.keys()) + add


class BaseComposite:
    "Base class for attr accesses in `self._extra_params` passed down to `self.components`"

    @property
    def _extra_params(self):
        if not hasattr(self, 'components'):
            self.components = []

        if type(self.components) not in {list, tuple}:
            self.components = [self.components]

        elif type(self.components) == tuple:
            self.components = list(self.components)

        args = []
        for component in self.components:
            args.extend([o for o in dir(component)
                         if not o.startswith('_')])

        return args

    def __getattr__(self, k):
        if k in self._extra_params:
            for component in self.components:
                if hasattr(component, k):
                    return getattr(self.components, k)

        raise AttributeError(k)

    def __dir__(self):
        return custom_dir(self, self._extra_params)
