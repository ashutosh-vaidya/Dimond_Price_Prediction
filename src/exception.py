import sys
from src.logger import logging


def error_message_detail(error, error_detail: sys):
    """
    The function "error_message_detail" returns a formatted error message with the file name, line
    number, and error message.

    Parametes:

    error -- the exception object that was raised.

    error_detail -- used to handle and display detailed error messages
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message

# Defines a custom exception that takes an error message and error detail as arguments
# and returns a formatted error message.
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
