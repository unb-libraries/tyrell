# tyrell
<p align="center">
<img src="assets/image.png" alt="drawing" width="400"/>
</p>

## Introduction
Tyrell summarizes documents.

## Documentation
Documentation is available [in the documentation folder](./documentation/README.md "Project Documentation").

## CLI Commands
### Start API Server
Start the API server:

```
poetry run api:start
```

### Summarize a document
#### (Requires: API Server Running)
The input file path should be an absolute path to a plaintext file you want to summarize. Extracting text from other formats is better handled by other tools.

```
poetry run summarize <file_path>
```
Examples:

```
poetry run summarize /home/jake/history_of_canada.txt > summary.json
```

## License
- As part of our 'open' ethos, UNB Libraries licenses its applications and workflows to be freely available to all whenever possible.
- Consequently, this repository's contents [unb-libraries/tyrell.lib.unb.ca] are licensed under the [MIT License](http://opensource.org/licenses/mit-license.html). This license explicitly excludes:
   - Any generated content remains the exclusive property of its author(s).
   - The UNB logo and associated suite of visual identity assets remain the exclusive property of the University of New Brunswick.