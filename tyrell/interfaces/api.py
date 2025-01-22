"""Provides the core API server for Tyrell."""
import requests
import socket
import sys

from filelock import FileLock
from flask import current_app, Flask, g, request, Response
from logging import Logger
from os import makedirs
from transformers import AutoTokenizer
from waitress import serve as waitress_serve

from tyrell.core import get_logger
from tyrell.core.builders import build_summarizer_chains

from tyrell.core.config import get_api_host, get_api_path, get_api_llm_config, get_api_port, get_gpu_lockfile, get_max_chunk_token_length, get_max_final_summary_context_tokens, get_data_dir
from tyrell.core import json_dumper
from tyrell.core.time import cur_timestamp, time_since
from tyrell.core.utils import report_memory_use, clear_gpu_memory
from tyrell.llm import LLM
from tyrell.llm.summarizer import summarize_document

CMD_STRING = 'api:start'

app = Flask(__name__)
logger = get_logger()
gpu_lock = FileLock(get_gpu_lockfile())

@app.before_request
def before_request():
    """Set the start time for the request."""
    g.start = cur_timestamp()

@app.route("/")
def default():
    """Default endpoint."""
    return "Endpoint Disabled."

@app.route(get_api_path(), methods=['POST'])
def summarize():
    """Summarize a document."""
    data = request.json
    document = data.get('document')
    debug = data.get('debug', False)

    gpu_request_lock_start = cur_timestamp()
    with gpu_lock:
        clear_gpu_memory()
        gpu_lock_wait_time = time_since(gpu_request_lock_start)
        logger.info("GPU lock acquired after %s seconds.", gpu_lock_wait_time)

        logger.info("Loading LLM...")
        llm_config = get_api_llm_config()
        llm_model_load_start = cur_timestamp()
        llm = LLM(logger, llm_config).get()
        llm_model_load_time = time_since(llm_model_load_start)

        logger.info("Building LLM Chains...")
        chain_build_start = cur_timestamp()
        chains = build_summarizer_chains(llm)
        chain_build_time = time_since(chain_build_start)

        logger.info("Initializing Tokenizer...")
        tokenizer_load_start = cur_timestamp()
        tokenizer = AutoTokenizer.from_pretrained(llm_config['tokenizer_repo'])
        tokenizer_load_time = time_since(tokenizer_load_start)

        max_chunk_token_length = get_max_chunk_token_length()
        max_final_summary_context_tokens = get_max_final_summary_context_tokens()
        summary = summarize_document(
            document,
            chains,
            tokenizer,
            max_chunk_token_length,
            max_final_summary_context_tokens,
            logger
        )
        # unset the model to free up memory
        llm = None

    summary['llm'] = {}
    summary['llm']['config'] = get_api_llm_config()
    summary['gpu_lock_wait_time'] = gpu_lock_wait_time
    summary['llm_model_load_time'] = llm_model_load_time
    summary['chain_build_time'] = chain_build_time
    summary['tokenizer_load_time'] = tokenizer_load_time
    summary['total_request_time'] = time_since(g.start)

    summary['generated_at'] = cur_timestamp()
    summary['agent'] = 'tyrell'
    summary['version'] = '1.0.0'

    if not debug:
        summary.pop('results', None)

    write_response_data(summary)

    if 'error' in summary:
        if "Empty document." in summary['error']:
            logger.error("Empty document.")
            return Response(json_dumper(summary, pretty=False), status=400, mimetype='application/json')
        else:
            logger.error("Error summarizing document:")
            logger.error(summary['error'])
            return Response(json_dumper(summary, pretty=False), status=500, mimetype='application/json')    

    return Response(json_dumper(summary, pretty=False), status=200, mimetype='application/json')

def start() -> None:
    """Starts the API server."""
    report_memory_use(logger)
    logger.info("Starting API server...")
    waitress_serve(app, host=get_api_host(), port=get_api_port())

def check_api_server_exit(log: Logger):
    """Exits if the API server is not running."""
    if not api_server_up():
        log.error("API server not running")
        sys.exit(1)

def api_server_up() -> bool:
    """Checks if the API server is running."""
    a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    location = (get_api_host(), get_api_port())
    return a_socket.connect_ex(location) == 0

def write_response_data(data: dict) -> None:
    """Writes the response data to a file."""
    data_dir = get_data_dir()
    summary_response_dir = f"{data_dir}/tyrell_responses"
    makedirs(summary_response_dir, exist_ok=True)
    final_filepath = f"{summary_response_dir}/response_{cur_timestamp()}.json"
    with open(final_filepath, 'w') as f:
        f.write(json_dumper(data, pretty=True))