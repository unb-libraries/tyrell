from datetime import datetime

from tyrell.core.chunker import chunk_document
from tyrell.core.time import cur_timestamp, time_since

def summarize_document(document, chains, tokenizer, max_chunk_token_length, max_final_summary_context_tokens, logger):
    """Summarizes a document.

    Args:
        document (str): The document to summarize.
        chains (dict): The chains to use.
        tokenizer (Tokenizer): The tokenizer to use.
        max_chunk_token_length (int): The maximum token length for a chunk.
        max_final_summary_context_tokens (int): The maximum token length for the final summary context.
        logger (Logger): The logger.

    Returns:
        dict: The response data.
    """
    chunks = chunk_document(tokenizer, document, max_chunk_token_length)
    logger.info("Chunked document into %d chunks...", len(chunks))

    if len(chunks) == 0:
        return {
            'error':'Empty document.',
        }

    if len(chunks) == 1:
        return summarize_single_chunk(chunks[0], chains, logger)
    return summarize_multiple_chunks(chunks, tokenizer, chains, max_chunk_token_length, max_final_summary_context_tokens, logger)

def summarize_single_chunk(chunk, chains, logger):
    """Summarizes a document that fits into a single chunk size.

    Args:
        chunk (str): The chunk to summarize.
        chains (dict): The chains to use.
        logger (Logger): The logger.

    Returns:
        dict: The response data.
    """
    chain = chains['summarize_oneshot']
    response_data = empty_response()
    total_inference_time = 0

    logger.info("Querying LLM Oneshot...")
    inference_start = cur_timestamp()
    chain_response = chain.invoke({
        "date": datetime.today().strftime("%Y-%m-%d"),
        "document": chunk
    })
    inference_time = time_since(inference_start)
    total_inference_time += inference_time

    response_data = update_response_data(response_data, 1, inference_time, chain_response['text'])

    logger.info("Responding with %s...", chain_response)
    return response_data

def summarize_multiple_chunks(chunks, tokenizer, chains, max_chunk_token_length, max_final_summary_context_tokens, logger):
    """Summarizes a document that was split into multiple chunks.

    Args:
        chunks (list): The chunks to summarize.
        tokenizer (Tokenizer): The tokenizer to use.
        chains (dict): The chains to use.
        max_chunk_token_length (int): The maximum token length for a chunk.
        max_final_summary_context_tokens (int): The maximum token length for the final summary context.
        logger (Logger): The logger.

    Returns:
        dict: The response data.
    """
    num_chunks = len(chunks)
    response_data = empty_response()
    total_inference_time = 0
    prior_summary, raw_inference_results_len = "", 0
    raw_inference_results, assembled_summaries, inference_methods = [], [], []
    resummary_counter, resummarized, compressed = 0, False, False

    response_data['results'].append([])
    for i, chunk in enumerate(chunks):
        chunk_summary = summarize_chunk(chunk, chains, prior_summary, i + 1, num_chunks, resummary_counter, logger)
        total_inference_time += chunk_summary['inference_time']

        chunk_summary['id'] = build_summary_id(resummary_counter, i + 1)
        response_data['results'][resummary_counter].append(chunk_summary)
        raw_inference_results.append(chunk_summary["response"])

        chunk_tokens = tokenizer.tokenize(chunk_summary['response'])
        raw_inference_results_len += len(chunk_tokens)
        prior_summary = chunk_summary['response']
    assembled_summaries.append(write_summary_string_from_raw_inference_results(raw_inference_results))
    inference_methods.append("initial-summary")
    
    inference_methods, assembled_summaries, raw_inference_results, response_data, resummary_counter, total_inference_time, resummarized, compressed = resummarize(
        raw_inference_results,
        chains,
        raw_inference_results_len,
        max_final_summary_context_tokens,
        tokenizer,
        max_chunk_token_length,
        assembled_summaries,
        response_data,
        resummary_counter,
        total_inference_time,
        inference_methods,
        logger
    )

    final_summary = summarize_final(raw_inference_results, chains, logger)
    inference_methods.append("final-summary")
    total_inference_time += final_summary['inference_time']
    response_data = finalize_response_data(response_data, assembled_summaries, final_summary, total_inference_time, inference_methods, resummarized, compressed)
    return response_data

def resummarize(raw_inference_results, chains, raw_inference_results_len, max_final_summary_context_tokens, tokenizer, max_chunk_token_length, assembled_summaries, response_data, resummary_counter, total_inference_time, inference_methods, logger):
    """Resummarizes the document if the summary generated in the initial pass is larger than the maximum final summary context tokens.

    Args:
        raw_inference_results (list): The raw inference results.
        chains (dict): The chains to use.
        raw_inference_results_len (int): The length of the raw inference results.
        max_final_summary_context_tokens (int): The maximum token length for the final summary context.
        tokenizer (Tokenizer): The tokenizer to use.
        max_chunk_token_length (int): The maximum token length for a chunk.
        assembled_summaries (list): The assembled summaries.
        response_data (dict): The response data.
        resummary_counter (int): The resummary counter.
        total_inference_time (float): The total inference time.
        inference_methods (list): The inference methods.
        logger (Logger): The logger.

    Returns:
        tuple: The inference methods, assembled summaries, raw inference results, response data, resummary counter, total inference time, resummarized, compressed.
    """
    logger.info("Checking %d tokens...", raw_inference_results_len)
    resummarized = False
    compressed = False
    while raw_inference_results_len > max_final_summary_context_tokens:
        exceeds_factor = raw_inference_results_len / max_final_summary_context_tokens
        logger.info(
            "Length of context tokens exceeds maximum by a factor of %.2f times. Resummarizing...",
            float(exceeds_factor)
        )
        if exceeds_factor > 1.25:
            logger.info("Fully Resummarizing...", raw_inference_results_len)
            resummary_counter += 1
            response_data['results'].append([])

            resummarize_source = "\n\n".join(raw_inference_results)
            raw_inference_results = []
            chunks = chunk_document(tokenizer, resummarize_source, max_chunk_token_length)
            num_chunks = len(chunks)

            raw_inference_results_len = 0
            prior_summary = ""

            for i, chunk in enumerate(chunks):
                chunk_summary = summarize_chunk(chunk, chains, prior_summary, i + 1, num_chunks, resummary_counter, logger)
                total_inference_time += chunk_summary['inference_time']

                chunk_summary['id'] = build_summary_id(resummary_counter, i + 1)
                response_data['results'][resummary_counter].append(chunk_summary)
                raw_inference_results.append(chunk_summary["response"])

                chunk_tokens = tokenizer.tokenize(chunk_summary['response'])
                raw_inference_results_len += len(chunk_tokens)
                prior_summary = chunk_summary['response']
            assembled_summaries.append(write_summary_string_from_raw_inference_results(raw_inference_results))
            inference_methods.append("full-resummary")
            resummarized = True

        elif exceeds_factor > 1.10:
            logger.info("Compressing...", raw_inference_results_len)
            raw_inference_results_len = 0
            compression_inference_time = 0
            compressed_results = []

            for i, inference_result in enumerate(raw_inference_results):
                compressed_result = compress_result(inference_result, chains, logger)

                compression_inference_time += compressed_result['inference_time']
                total_inference_time += compressed_result['inference_time']
                result_tokens = tokenizer.tokenize(compressed_result['response'])
                raw_inference_results_len += len(result_tokens)

                compressed_results.append(compressed_result['response'])
            compressed = True

            response_data['compressions'].append({
                "source": raw_inference_results,
                "result": compressed_results,
                "inference_time": compression_inference_time,
            })
            assembled_summaries.append(write_summary_string_from_raw_inference_results(compressed_results))
            inference_methods.append("compression")
            raw_inference_results = compressed_results
        else:
            logger.info("Skipping Resummarization As Threshold is Near.")
            break
    return inference_methods, assembled_summaries, raw_inference_results, response_data, resummary_counter, total_inference_time, resummarized, compressed

def summarize_chunk(chunk, chains, prior_summary, cur_chunk_no, total_chunk_no, resummary_counter, logger):
    """Summarizes a single chunk.
    
    Args:
        chunk (str): The chunk to summarize.
        chains (dict): The chains to use.
        prior_summary (str): The prior summary.
        cur_chunk_no (int): The current chunk number.
        total_chunk_no (int): The total number of chunks.
        resummary_counter (int): The resummary counter.
        logger (Logger): The logger.
    
    Returns:
        dict: The response data.
    """
    chain = chains['summarize_chunk']
    logger.info("Summarizing Chunk %d.%d/%d.%d...", resummary_counter, cur_chunk_no, resummary_counter, total_chunk_no)
    inference_start = cur_timestamp()
    chunk_response = chain.invoke({
        "date": datetime.today().strftime("%Y-%m-%d"),
        "chunk": chunk,
        "prior_summary": prior_summary,
        "cur_chunk_no": cur_chunk_no,
        "total_chunk_no": total_chunk_no
    })
    inference_time = time_since(inference_start)
    formatted_response = chunk_response.strip()
    logger.info("Summarized Chunk %d.%d into: %s", resummary_counter, cur_chunk_no, formatted_response)
    return {
        "inference_time": inference_time,
        "response": formatted_response
    }

def compress_result(raw_inference_result, chains, logger):
    """Compresses a slightly-oversized context.

    Args:
        raw_inference_result (str): The raw inference result.
        chains (dict): The chains to use.
        logger (Logger): The logger.

    Returns:
        dict: The response data.
    """
    compression_chain = chains['compress']
    logger.info("Compressing result: %s...", raw_inference_result)
    logger.info("Querying LLM for compression...")
    inference_start = cur_timestamp()
    compressed_version = compression_chain.invoke({
        "date": datetime.today().strftime("%Y-%m-%d"),
        "original": raw_inference_result
    })
    inference_time = time_since(inference_start)
    logger.info("Compressed to: %s...", compressed_version)
    return {
        "inference_time": inference_time,
        "response": compressed_version.strip()
    }


def summarize_final(raw_inference_results, chains, logger):
    """Summarizes the final document from the section summaries.

    Args:
        raw_inference_results (list): The raw inference results.
        chains (dict): The chains to use.
        logger (Logger): The logger.
    
    Returns:
        dict: The response data.
    """
    final_chain = chains['summarize_final']
    full_summary_string = write_context_from_raw_inference_results(raw_inference_results)

    logger.info("Finalizing Summary from section summaries %s...", full_summary_string)
    inference_start = cur_timestamp()
    final_response = final_chain.invoke({
        "date": datetime.today().strftime("%Y-%m-%d"),
        "summary": full_summary_string
    })
    inference_time = time_since(inference_start)
    logger.info("Responding with %s...", final_response)
    return {
        "inference_time": inference_time,
        "response": final_response.strip()
    }

def update_response_data(response_data, chunk_id, inference_time, response_text):
    response_data['results'].append({
        "id": chunk_id,
        "inference_time": inference_time,
        "response": response_text
    })
    response_data['summaries'] = []
    response_data['summaries'].append({
        "id": chunk_id,
        "summary": response_text,
        "length": len(response_text)
    })
    response_data['summary'] = response_text
    response_data['total_inference_time'] = inference_time
    return response_data

def finalize_response_data(response_data, assembled_summaries, final_summary, total_inference_time, inference_methods, resummarized, compressed):
    response_data['summary_inference_time'] = final_summary['inference_time']
    response_data['summaries'] = []
    for i, assembled_summary in enumerate(assembled_summaries):
        response_data['summaries'].append({
            "id": i + 1,
            "summary": assembled_summary,
            "length": len(assembled_summary)
        })
    response_data['summaries'].append({
        "id": len(assembled_summaries) + 1,
        "summary": final_summary['response'],
        "length": len(final_summary['response'])
    })
    response_data['response'] = final_summary['response']

    response_data['resummarized'] = resummarized
    response_data['compressed'] = compressed
    response_data['inference_methods'] = inference_methods
    response_data['total_inference_time'] = total_inference_time

    return response_data

def write_context_from_raw_inference_results(raw_inference_results):
    summary_string = ""
    num_results = len(raw_inference_results)
    for i, inference_result in enumerate(raw_inference_results):
        summary_string += f"\n\nChunk ({i + 1} of {num_results}):\n{inference_result}"
    return summary_string

def write_summary_string_from_raw_inference_results(raw_inference_results):
    return "\n\n".join(raw_inference_results)

def empty_response() -> dict:
    """Returns an empty response."""
    return {"summaries": [], "compressions": [], "results": []}

def build_summary_id(resummary_counter: int, chunk_no: int) -> str:
    """Builds a summary ID."""
    return f"{resummary_counter}.{chunk_no}"
