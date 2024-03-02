# Class to handle the Logging for all modules in the same way.
# Additionally print always the same initial log message at the beginning

# Import the external packages
# Logging machanism
import logging
# Operating system functionalities
import os
# Stream handling
import io

class OwnLogging():
    """
    A own class implementation for log messages within the program, to handle it always in the same way.
    ----------
    Attributes:
    package_name : str
        The name of the package used by the logger.
        If no name is specified, the root logger is used.
    logger : Logger
        The logger, which can be used as a standard logger of package logging
    isLevelDebug : Boolean
        Gives the state if the logger level is DEBUG
    ----------
    Methods:
        no methods
    """

    # Constructor Method
    def __init__(self, package_name=None):
        # Get the logger, use the specific name if available
        if package_name is None:
            self.__logger = logging.getLogger()
        else:
            self.__logger = logging.getLogger(package_name)

        # Configure the logging mechanism
        # Use the basicConfig when writing to one file 
        #logging.basicConfig(
        #            # Default Level 
        #            level=logging.INFO,
        #            # Write the logs into file 
        #            filename=os.path.join("logs","main.log"),filemode="w",
        #            # Formatting the logs
        #            format="\t%(asctime)s,%(msecs)d %(levelname)-8s [%(name)s:%(filename)s-%(funcName)s(%(lineno)d)] \n%(message)s"
        #)
        # When writing to different files:
        # Formatting the logs
        formatter = logging.Formatter("\t%(asctime)s,%(msecs)d %(levelname)-8s [%(name)s:%(filename)s-%(funcName)s(%(lineno)d)] \n%(message)s")
        # Write the logs into file
        log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(os.path.join(log_dir, package_name + ".log"), 'w')
        file_handler.setFormatter(formatter)
        self.__logger.addHandler(file_handler)
        # Set the level
        self.__logger.setLevel(logging.INFO)

        self.__logger.debug("Logger initialization for package/module <%s>", package_name)

    @property
    def logger(self):
        """
        The method is declared as property (decorated as getter)
        The logger, which can be used as a standard logger of package logging
        """
        return self.__logger

    @property
    def isLevelDebug(self):
        """
        The method is declared as property (decorated as getter)
        Boolean, if the logger level is DEBUG.
        """
        return True if self.__logger.getEffectiveLevel() == logging.DEBUG else False

#########################################################

def log_overview_data_frame(logger, df):
    """
    Logging some information about the data frame
    ----------
    Parameters:
        logger : Logger
                The Logger to log with
        df : pandas.core.frame.DataFrame
                The data
    ----------
    Returns:
        no returns
    """

    # Print the first 5 rows
    logger.info("Data Structure (first 5 rows): %s", df.head(5))
    # Print some information (pipe output of DataFrame.info to buffer instead of sys.stdout for correct logging)
    buffer = io.StringIO()
    buffer.write("Data Info: ")
    df.info(buf=buffer)
    logger.info(buffer.getvalue())
