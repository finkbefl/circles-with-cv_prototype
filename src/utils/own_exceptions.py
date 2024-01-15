# Own Exception Classes

# A own exception class when a illegal argument is passed,
# inherit from ValueError as it is as close to what this exception is intended for
class IllegalArgumentError(ValueError):
    """
    Exception raised for errors when passing an illegal argument
    ----------
    Attributes:
        message : str
            message -- explanation of the error
    ----------
    Methods:
        no methods
    """
    def __init__(self, message="This argument is not allowed!"):
        self.message = message
        super().__init__(self.message)

# A own exception class for errors when trying to query a database table,
# inherit from AttributeError as it is as close to what this execption is intended for
class DatabaseQueryError(AttributeError):
    """
    Exception raised for errors when query a database table
    ----------
    Attributes:
        message : str
            message -- explanation of the error
    ----------
    Methods:
        no methods
    """
    def __init__(self, message="This database query is not allowed!"):
        self.message = message
        super().__init__(self.message)

# A own exception class for errors when trying to calc with two datasets which not share the same x vals
# inherit from IndexError as it is as close to what this execption is intended for
class XValNotMatch(IndexError):
    """
    Exception raised for errors when calculating with two datasets which not share the same x values
    ----------
    Attributes:
        message : str
            message -- explanation of the error
    ----------
    Methods:
        no methods
    """
    def __init__(self, message="X values of datasets not matching!"):
        self.message = message
        super().__init__(self.message)