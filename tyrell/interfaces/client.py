"""Provides a command to summarize a document."""
import os
import sys

import requests
from logging import Logger

from tyrell.core.config import get_client_uri, get_client_timeout, get_client_keypair, get_client_user_agent
from tyrell.core import get_logger
from tyrell.interfaces.api import check_api_server_exit
from tyrell.core import json_dumper

CMD_STRING = 'summarize'

def summarize(args: list=sys.argv) -> None:
    """Summarizes the provided document.

    Args:
        args (list, optional): The arguments for the summary. Defaults to sys.argv.
    """
    log = get_logger()
    check_api_server_exit(log)
    validate_args(args, log)

    with open(args[1], 'r') as f:
        document = f.read()

    uri = get_client_uri()
    log.info("Querying %s...", uri)

    keypair = get_client_keypair()

    headers = {
        "x-pub-key": keypair[0],
        "x-api-key": keypair[1],
    }

    json_query = {
        "client": get_client_user_agent(),
        "document": document
    }

    r = requests.post(
        uri,
        json=json_query,
        headers=headers,
        timeout=get_client_timeout()
    )

    if r.status_code != 200:
        log.error("Failed to summarize document:")
        log.error(r.text)
        sys.exit(1)

    print(
        json_dumper(r.json())
    )

def validate_args(args: list, log: Logger) -> None:
    """Validates the arguments for the command and exits if invalid.

    Args:
        args (list): The arguments to validate.
        log (Logger): The logger to use.
    """
    if len(args) < 1:
        log_usage(log)
        sys.exit(1)

    try:
        if args[1] == "":
            raise ValueError
    except Exception:
        log.warning("File Path cannot be empty")
        log_usage(log)
        sys.exit(1)
    
    # does the file exist at the provided path?
    if not os.path.exists(args[1]):
        log.warning("File does not exist")
        log_usage(log)
        sys.exit(1)

def log_usage(log: Logger) -> None:
    """Outputs the usage for the command.

    Args:
        log (Logger): The logger to use.
    """
    log.warning("Usage: poetry run %s <filepath>", CMD_STRING)
