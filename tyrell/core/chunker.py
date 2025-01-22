def chunk_document(tokenizer, document: str, max_token_length: int) -> list:
    """
    Splits a document into chunks where each chunk does not exceed the max token length.

    Parameters:
        document (str): The input document as a string.
        max_token_length (int): The maximum token length for each chunk.
        model_file (str): The path to the SentencePiece model file (.gguf).

    Returns:
        list: A list of document chunks as strings.
    """
    # Tokenize the document using SentencePiece
    tokens = tokenizer.tokenize(document)
    num_tokens = len(tokens)

    optimal_token_length = find_nearest_equal_token_size(num_tokens, max_token_length)

    chunks = []
    current_chunk = []
    current_length = 0

    for i, token in enumerate(tokens):
        # If adding the current token would exceed the optimal token length, start a new chunk
        if current_length + len(token) > optimal_token_length:
            chunks.append(tidy_sentencepiece_output(''.join(current_chunk)))
            current_chunk = []
            current_length = 0

        current_chunk.append(token)
        current_length += len(token)

        # If this is the last token, add the current chunk
        if i == num_tokens - 1:
            chunks.append(tidy_sentencepiece_output(''.join(current_chunk)))

    return chunks

def find_nearest_equal_token_size(num_tokens: int, max_token_length: int) -> int:
    """
    This function finds the optimal number of tokens that will produce equal length chunks.

    A situation may arise when if using the max token length, the last chunk could be very short. We should
    instead find an optimal number that will produce equal length chunks.

    Parameters:
        num_tokens (int): The number of tokens in the original documents.
        max_token_length (int): The maximum token length.

    Returns:
        int: The number of tokens that will produce as equal length chunks as possible.
    """
    # Calculate the number of chunks needed
    num_chunks = (num_tokens + max_token_length - 1) // max_token_length
    return num_tokens // num_chunks
    
def tidy_sentencepiece_output(output: str) -> str:
    """
    Cleans up the output of SentencePiece tokenization.

    Parameters:
        output (str): The SentencePiece tokenization output.

    Returns:
        str: The cleaned up output.
    """
    return output.replace('▁', ' ').replace('<0x0A>', '\n')
