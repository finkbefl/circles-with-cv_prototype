# Functions to check input parameter

# Import internal packages/ classes
from utils.own_exceptions import IllegalArgumentError

def checkParameter(parameter,  type, noneIsValid=False):
    """
    Check if the given parameter is not None and match the given type
    ----------
    Parameters:
        parameter
            The parameter to check
        noneIsValid : Boolean
            True, if None is a valid state
        type : type
            The type of the parameter
    ----------
    Returns:
        no return
    """
    if not noneIsValid and parameter is None:
        raise IllegalArgumentError("The parameter cannot be None")
    # It must be the given type
    elif parameter is not None and not isinstance(parameter, type):
        raise TypeError("The parameter does not match the given type")

def checkParameterString(string):
    """
    Check if the given string is not None, not empty and match the string type
    ----------
    Parameters:
        string : str
            The str parmeter to check
    ----------
    Returns:
        no return
    """
    if string == "":
        raise IllegalArgumentError("The parameter cannot be empty")
    else:
        checkParameter(string, str)