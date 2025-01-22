"""Provides standard prompts for the LLM."""

def get_summarize_oneshot() -> str:
    return """<#meta#>
    - Date: {date}
    - Task: summary
    <#system#>
    Your main objective is to condense the content of the document into a concise summary, capturing the main points and themes.
    <#chat#>
    <#user#>
    Please read the provided Original section to understand the context and content. Use this understanding to generate a summary of the Original section. Separate the article into chunks, and sequentially create a summary for each chunk. Focus on summarizing the Original section, ignoring any details about sponsorships/advertisements in the text.

    Summarized Sections:
    1. For each chunk, provide a concise summary. Start each summary with "Chunk (X of Y):" where X is the current chunk number and Y is the total number of chunks.

    To craft a Final Summary:
    1. Read the Summarized Sections: Carefully review all the summarized sections you have generated. Ensure that you understand the main points, key details, and essential information from each section.
    2. Identify Main Themes: Identify the main themes and topics that are prevalent throughout the summarized sections. These themes will form the backbone of your final summary.
    3. Consolidate Information: Merge the information from the different summarized sections, focusing on the main themes you have identified. Avoid redundancy and ensure the consolidated information flows logically.
    4. Preserve Essential Details: Preserve the essential details and nuances that are crucial for understanding the document. Consider the type of document and the level of detail required to capture its essence.
    5. Draft the Final Summary: After considering all the above points, draft a final summary that represents the main ideas, themes, and essential details of the document. Start this section with "Final Summary:"

    Ensure that your final output is thorough, and accurately reflects the document's content and purpose.
    <#user_context#>
    Original:
    {document}
    <#bot#>"""

def get_compress_result() -> str:
    return """<#meta#>
    - Date: {date}
    - Task: condense
    <#system#>
    Your main objective is to condense the content of the document slightly, retaining all key points and avoiding unnecessary reductions.
    <#chat#>
    <#user#>
    Please read the provided Original section to understand the context and content. Use this understanding to generate a slighly condensed version of the Original section, incorporating relevant details and maintaining coherence.
    <#user_context#>
    Original:
    {original}

    Slightly Condensed Version:
    <#bot#>"""

def get_summarize_chunk() -> str:
    return """<#meta#>
    - Date: {date}
    - Task: summary
    <#system#>
    Your main objective is to condense the content of the document into a concise summary, capturing the main points and themes.
    <#chat#>
    <#user#>
    Please read the provided Original section to understand the context and content. Use this understanding to generate a summary of the Original section, incorporating relevant details and maintaining coherence with the Prior Summary.

    Notes:
    - The Prior Summary was created from the chunk of the document directly preceding this chunk.
    - Ignore the details already included in the Prior Summary when creating the new Summary.
    - Focus on summarizing the Original section, taking into account the context provided by the Prior Summary.
    - Ignore any details about sponsorships/advertisements in the text.
    <#user_context#>
    Prior Summary:
    {prior_summary}

    Original (Chunk {cur_chunk_no} of {total_chunk_no} total):
    {chunk}

    Summary (Chunk {cur_chunk_no} of {total_chunk_no} total):
    <#bot#>"""

def get_summarize_final() -> str:
    return """<#meta#>
    - Date: {date}
    - Task: summary
    <#system#>
    Your main objective is to condense the content of the document into a concise summary, capturing the main points and themes.
    <#chat#>
    <#user#>
    To craft a Final Summary:

    1. Read Summarized Sections: Carefully review all the summarized sections of the document. Ensure that you have a clear understanding of the main points, key details, and essential information presented in each section.
    2. Identify Main Themes: As you go through the summarized sections, identify the main themes and topics that are prevalent throughout the document. Make a list of these themes as they will form the backbone of your final summary.
    3. Consolidate Information: Merge the information from the different summarized sections, focusing on the main themes you have identified. Avoid redundancy and ensure that the consolidated information flows logically.
    4. Preserve Essential Details: While consolidating, ensure that you preserve the essential details and nuances that are crucial for understanding the document. Consider the type of document and the level of detail required to accurately capture its essence.
    5. Check for Completeness: After drafting the final summary, review it to ensure that it accurately represents the main ideas, themes, and essential details of the document.

    Please remember to be thorough, and ensure that the final summary is a true reflection of the document's content and purpose.
    <#user_context#>
    Summarized Sections:
    {summary}

    Concise Summary:
    <#bot#>"""
