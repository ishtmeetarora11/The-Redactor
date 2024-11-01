# cis6930fa24-project1

Name: Ishtmeet Singh Arora

## Project Description
This project automates the redaction of sensitive information from text documents, such as police reports, court transcripts, and hospital records. The program processes input text files, identifies sensitive information using Natural Language Processing (NLP) techniques and regular expressions, and redacts this information to prevent the disclosure of confidential data. The sensitive information includes names, dates, phone numbers, addresses, and specific concepts.

The project involves:

Fetching Input Data: Reading text files from a specified input directory or matching a given glob pattern.

Extracting Relevant Information: Using NLP models (SpaCy and Hugging Face Transformers) and custom regex patterns to detect sensitive entities.

Redacting Sensitive Data: Replacing sensitive information with a redaction character while preserving the original text format and length.

Storing Redacted Data: Writing the redacted text to an output directory for further use or analysis.

Generating Statistics: Summarizing the redaction process by counting the types and instances of redacted entities.

## How to install

```
pipenv install -e .
```

## How to run
Run the following command on terminal

``` bash
python main.py --input <input_file_glob> --output <output_directory> [options]
```

Example:

``` bash
pipenv run python redactor.py --input '*.txt' \
                    --names --dates --phones --address\
                    --concept 'kids' \
                    --output 'files/' \
                    --stats stderr
```

## Running Test Cases
```bash
pipenv run pytest
```


## Functions in Redactor.py

### main()

The main function initiates the redaction process by:

Parsing Command-Line Arguments: Defines input files, output directory, entity types to redact, and where to write statistics.

Processing Files: Loops through the input files and applies redaction based on the specified parameters.

Outputting Statistics: Writes redaction counts to the chosen destination (stdout, stderr, or file).

### initialize_spacy_nlp()

```
def initialize_spacy_nlp():

    Initializes a SpaCy NLP pipeline with custom entity recognition patterns for redacting names, dates, phone numbers, and addresses. Uses lazy loading to load the model only once.
        
    Returns:
        A SpaCy NLP pipeline configured with custom patterns.

```

### initialize_hf_pipeline()

```
def initialize_hf_pipeline():

    Initializes a Hugging Face NER pipeline using the dslim/bert-base-NER model. This pipeline is used for entity detection in the redaction process.
        
    Returns:
        A Hugging Face pipeline object for Named Entity Recognition (NER).

```
### merge_overlapping_spans(spans)

```
def merge_overlapping_spans(spans):

    Merges overlapping or adjacent character spans to ensure there are no redundant redactions in overlapping areas.
        
    Args:
        spans (list of tuples): List of character index ranges to merge.

    Returns:
        A list of merged spans.
        
```
### identify_concept_sentences(text, concepts)

```
def identify_concept_sentences(text, concepts):

    Identifies sentences containing specified concepts (e.g., "confidential") for redaction.

    Args:
        text (str): The text to analyze.

        concepts (list of str): List of keywords or phrases to identify.
        
    Returns:
        List of character index ranges for sentences containing any specified concepts.

```

### redact_entities_hf(text, targets, stats)

```
def redact_entities_spacy(text, targets, stats):

    Uses Hugging Face’s NER pipeline to identify and redact specified entities.

    Args:
        text (str): Text to be redacted.

        targets (list of str): List of entities to redact (e.g., ['names', 'addresses']).

        stats (dict): Dictionary tracking redaction counts.
        
    Returns:
        List of character index ranges to redact.

```

### redact_entities_spacy(text, targets, stats)

```
def redact_entities_spacy(text, targets, stats):

    Uses the SpaCy pipeline to identify and redact specific entities in the text based on target categories (e.g., names, dates).

    Args:
        text (str): Text to be redacted.

        targets (list of str): List of entities to redact (e.g., ['names', 'dates']).

        stats (dict): Dictionary tracking redaction counts.
        
    Returns:
        List of character index ranges to redact.

```

### redact_email_headers(text, targets, stats)

```
def redact_email_headers(text, targets, stats):

    Redacts names found in email headers (such as 'From', 'To', 'Cc') within the text.

    Args:
        text (str): Text to be redacted.

        targets (list of str): List of entity types to redact (e.g., ['names']).

        stats (dict): Dictionary tracking redaction counts.
        
    Returns:
        List of character index ranges to redact.

```

### redact_entities_regex(text, targets, stats)

```
def redact_entities_regex(text, targets, stats):

    Uses regular expressions to identify and redact specified entities, such as phone numbers, dates, and names, based on pattern matching.

    Args:
        text (str): Text to be redacted.

        targets (list of str): List of entities to redact (e.g., ['phones', 'dates']).

        stats (dict): Dictionary tracking redaction counts.
        
    Returns:
        List of character index ranges to redact.

```

### write_stats(stats, destination)

```
def write_stats(stats, destination):

    Outputs the redaction statistics to the specified destination (stderr, stdout, or a file path).

    Args:
        stats (dict): Dictionary containing counts of redacted items by category.

        destination (str): Output destination for statistics.

```

### process_file(file_path, args, stats)

```
def write_stats(stats, destination):

    Processes a single text file, applying redaction based on specified arguments, and saves the redacted content.

    Args:
        file_path (str): Path of the input file.

        args (Namespace): Parsed command-line arguments with options for redaction.
        
        stats (dict): Dictionary to accumulate redaction statistics.

```

## Bugs and Assumptions

### Assumptions

-> The incident data is assumed to be available in a PDF format at the provided URL. The code relies on this assumption, as it uses a PDF parser (PdfReader) to extract content.

-> The create DB function assumes that we want to start with a clean database each time the script is run. It deletes the existing database (normanpd.db) if it already exists.

-> The connection passed to the function is assumed to be a valid SQLite connection object.


## Test Cases

### test_db.py

This tests the functionality of a database-related module (db) using the pytest framework and mocking (unittest.mock). It ensures that the database is created, populated, and can produce a status summary, all without writing to disk (by using an in-memory SQLite database).

The patch() function is used to mock sqlite3.connect so that when the db.createdb() function is called, it creates an in-memory database instead of a file-based one.

The db.createdb() function is called, which creates the database schema (table) and returns a connection object (conn).

After the test using this fixture is completed, the connection is closed (conn.close()).

The connection object is then passed to the test functions (test_createdb, test_populatedb, and test_status).

The test checks that the createdb() function successfully creates the incidents table in the database.

The test verifies that the populatedb() function correctly inserts data into the incidents table.

The test checks that the status() function correctly generates a summary of incidents by nature (e.g., "Theft|1", "Assault|1") and prints it.

### test_extractincidents.py

This test case is designed to verify that the extractIncidents function from the extractincidents module correctly processes a PDF containing incident records and extracts the relevant data into structured dictionaries.

The mock PDF content is structured to resemble the layout of the actual incident PDF, with headers like Date / Time, Incident Number, Location, Nature, and Incident ORI. Two sample incidents are included

The PDF data is read and stored in the variable info

The extractIncidents function is called with the fetched PDF data (info). The function processes the binary data, extracts incident information, and returns it as a list of dictionaries.

Two assertions are made to ensure that the data is correctly extracted


### test_incident.py

This test case is designed to validate the behavior of the fetchIncidents function by mocking the network request (urllib.request.urlopen). It ensures that the function properly sends the HTTP request, sets the appropriate headers, and handles the response

The @patch('urllib.request.urlopen') decorator is used to replace the actual urlopen function with a mock object. This allows the test to simulate the behavior of a real HTTP request without actually making a network call.

The mock object (mock_urlopen) is passed as an argument to the test function.

The test uses http://example.com/test as the sample URL, which simulates the URL that the fetchIncidents function will use to fetch data.

The test asserts that the value returned by fetchIncidents matches the expected value b"Ishtmeet", which was set in the mock response.

