import functools
import logging

from flask import request


def log_request_and_response(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Log request information
        logging.info(f"Received {request.method} request to {request.path}")

        # Call the original function
        response = func(*args, **kwargs)

        # Log response information
        return response

    return wrapper
