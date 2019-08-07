import abc
import six
from collections import OrderedDict

REGISTERED_STRATEGIES = OrderedDict()


class RegistrationError(ValueError):

    def __init__(self, cls_name, subclass_type):
        msg = "Class %s is not an abstract class, yet it's subclass type " \
              "was specified as %s. All registered subclasses must set the " \
              "class parameter `subclass_type` to be properly registered !" % (
               cls_name, subclass_type)
        ValueError.__init__(self, msg)


class RegisteredStrategy(abc.ABCMeta):

    def __new__(mcls, class_name, bases, class_dict):
        name = class_name
        class_dict['name'] = name

        cls = super(RegisteredStrategy, mcls).__new__(mcls, class_name, bases, class_dict)
        subclass_type = cls.subclass_type

        if 'abstract' not in name.lower() and subclass_type == 'abstract':
            raise RegistrationError(name, subclass_type)

        # override the subclass type if it is an Abstract class.
        if 'abstract' in name.lower():
            subclass_type = 'abstract'

        # register the class object
        if subclass_type in REGISTERED_STRATEGIES:
            if name in REGISTERED_STRATEGIES[subclass_type]:
                raise RuntimeError("A Strategy has already been registered with the given "
                                   "class name in this subclass type. "
                                   "Set a different name for this class.")

            else:
                REGISTERED_STRATEGIES[subclass_type][name] = cls

        else:
            # create a new subclass type in the registry, and add the registered subclass itself
            REGISTERED_STRATEGIES[subclass_type] = OrderedDict()
            REGISTERED_STRATEGIES[subclass_type][name] = cls

        return cls


@six.add_metaclass(RegisteredStrategy)
class AbstractStrategy(object):
    # Subclass type designates which registry the class will belong to.
    subclass_type = 'abstract'

    def __init__(self,):
        super(AbstractStrategy, self).__init__()


def get(strategy_name):
    strategy_list = []
    for subclass_type in REGISTERED_STRATEGIES.keys():

        if strategy_name in REGISTERED_STRATEGIES[subclass_type]:
            return REGISTERED_STRATEGIES[subclass_type][strategy_name]

        # strategy not found, add the currently registered registry
        strategy_list.extend(list(REGISTERED_STRATEGIES[subclass_type].keys()))

    raise RuntimeError("Strategy %s not found in registered registry ! "
                       "Found registry = %s" % (strategy_name,
                                                str(strategy_list)))
