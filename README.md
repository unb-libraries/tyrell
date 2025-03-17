# tyrell
<p align="center">
<img src="assets/image.png" alt="drawing" width="400"/>
</p>

## Introduction
Tyrell summarizes documents by leveraging modern LLMs.

## Documentation
Some stub documentation is currently available [in the documentation folder](./documentation/README.md "Project Documentation").

## Pre-requisites
- Python 3.8+
- [Poetry](https://python-poetry.org/docs/)
- [Nvidia drivers](https://www.nvidia.com/Download/index.aspx)
- [CUDA](https://developer.nvidia.com/cuda-downloads)

## CLI Commands
### `api:start`

Start the API server:

`poetry run api:start`

## Client Request Format
The client should send a POST request to the API server with the following data format:

```json
{
  "text": "The text you want to summarize"
}
```

## Convenience Commands
Other commands are available for convenience:

### `summarize`
#### (API Server Must be Running)

The input file path should be an absolute path to a plaintext file you want to summarize. Extracting text from other formats is better handled by other tools.

'poetry run summarize <file_path>'

As an example, to summarize Melville's Moby Dick (Obtained from [project Gutenburg](https://www.gutenberg.org/files/2701/old/moby10b.txt))

```
poetry run summarize ./moby10b.txt > summary.json
```

## License
- In line with our 'open' ethos, UNB Libraries makes its applications and workflows freely available to everyone whenever possible.
- As a result, the contents of this repository [unb-libraries/tyrell] are licensed under the [MIT License](http://opensource.org/licenses/mit-license.html). This license explicitly excludes:
   - Any content that remains the exclusive property of its author(s).
   - The UNB logo and any associated visual identity assets remain the exclusive property of the University of New Brunswick.