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

### Command-Line Arguments ( Parameters )

```
    --input: Glob pattern to specify input files.
    --output: Directory to save redacted files.
    --names: Redacts names from the text. (Name of only Persons/People)
        Description: Redacts names of people mentioned in the text. This includes both first and last names, names with initials, and commonly capitalized names.

        Detection Method:
        SpaCy’s Named Entity Recognition (NER) model, which recognizes PERSON entities.
        Custom regex patterns for capitalized words in email headers and names in common formats.

    --dates: Redacts dates from the text.
        Description: Redacts any written date in various common formats, such as 4/9/2025, April 9th, 2025, or 22-02-2022.

        Detection Method:
            Custom regex patterns for multiple date formats.
            SpaCy’s NER model, which detects DATE entities.

    --phones: Redacts phone numbers from the text.
        Description: Redacts phone numbers in various formats, including international, local, and common representations.

        Detection Method:
            Custom regex patterns to capture phone numbers in formats like (123) 456-7890, 123-456-7890, +1 123-456-7890, etc.

    --address: Redacts addresses from the text.
        Description: Redacts physical addresses, typically street addresses, that may include street names, city names, and postal codes.

        Detection Method:
            Custom regex patterns to detect common address structures.
            SpaCy’s NER model to detect GPE (Geo-Political Entity) and LOC (location) entities.

    --concept: Specifies concepts to redact in sentences.
        Description: Redacts sentences or paragraphs containing specific "concepts." A concept represents an idea or theme that may be sensitive.

        Parameter: Accepts one or more words or phrases as arguments. Each word or phrase represents a concept that needs to be redacted.

        Context Definition: A "concept" is defined as a word or theme and its semantic associations. For example, the concept of prison might include similar terms like jail, incarceration, or detention.

        The program uses regular expressions to find sentences containing exact matches of the provided concept words.
        SpaCy's NLP model helps identify sentences where contextually similar terms (e.g., prison and jail) may occur in the same semantic context.

    --stats: Specifies where to output redaction statistics (stderr, stdout, or a filepath).

        Description: Outputs a summary report showing the total count of each type of redacted entity across all processed files. This flag specifies the destination for this report, which can be set to stdout, stderr, or a specific file path.

        Parameters:
            Accepts a single argument to define where the statistics report should be sent:
            stdout - Prints the report to the standard output.
            stderr - Prints the report to the standard error stream.
            file path - Saves the report to a specified file path.
        
        output format:
            Names redacted: 
            Dates redacted: 
            Phone numbers redacted: 
            Addresses redacted: 
            Concepts redacted: 

        The write_stats function aggregates and writes the total counts of all redacted entities from all input files.
        If an error occurs while writing to a file path (e.g., if the path is invalid), an error message is printed to stderr.

```


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

    This function accumulates counts for all redacted items across multiple files. If a redaction span is identified multiple times by different methods (e.g., SpaCy, Hugging Face, and regex)

```

### process_file(file_path, args, stats)

```
def process_file(file_path, args, stats):

    Processes a single text file, applying redaction based on specified arguments, and saves the redacted content.

    Args:
        file_path (str): Path of the input file.

        args (Namespace): Parsed command-line arguments with options for redaction.

        stats (dict): Dictionary to accumulate redaction statistics.

```

## Bugs and Assumptions

### Assumptions

-> The redact_entities_regex function may not always recognize names accurately, especially in cases of uncommon names or names with special characters. It relies on capitalized words, which could lead to false positives (e.g., capitalized words in sentences being treated as names).

-> The hardcoded patterns for dates, phone numbers, and addresses may not cover all possible formats encountered in real-world data. For instance, international phone number formats are not fully supported.

-> The write_stats function directly increments redaction counts without checking for duplicates. This might cause inaccurate counts, especially if an entity is identified multiple times by different models.


## Test Cases

### test_address.py

test_redact_address_spacy: Tests the redact_entities_spacy function by providing a sample address. Asserts that no address is detected to check if the function works correctly with a non-matching target.

test_redact_address_regex: Tests the redact_entities_regex function by providing an address in text format. Asserts that one address is detected using regex.


### test_concepts.py

test_identify_concept_sentences: Tests the identify_concept_sentences function by providing text with specific concepts. Asserts that sentences containing specified concepts are correctly identified.


### test_dates.py

test_redact_dates_spacy: Tests the redact_entities_spacy function to detect dates in the text. Asserts that one date is detected using SpaCy.

test_redact_dates_regex: Tests the redact_entities_regex function to detect multiple dates in different formats. Asserts that two dates are correctly identified using regex.


### test_identify_concepts.py

test_no_concepts: Verifies that no concepts are detected if the text does not contain specified keywords.

test_single_concept: Checks if a single concept is identified and redacted correctly in the text.

test_multiple_concepts: Tests multiple concept detection within the text and ensures the correct spans are returned.

test_concepts_case_insensitive: Verifies that the function is case-insensitive by detecting concepts written in different cases.

test_concepts_partial_match: Ensures that only exact matches are detected, ignoring partial matches within words.


### test_main.py

test_main_success: Mocks file processing to ensure main function correctly processes multiple files with specified options and calls the appropriate functions.

test_main_no_files_matched: Verifies that an error message is displayed when no files match the input pattern.

test_main_with_single_file: Tests the main function with a single file and verifies that processing functions are called correctly.


### test_merge_spans.py

test_no_overlaps: Tests that non-overlapping spans are returned as-is without modification.

test_with_overlaps: Ensures overlapping spans are merged into a single span.

test_adjacent_spans: Verifies that adjacent spans are merged into one span.

test_nested_spans: Checks if nested spans are merged into a single span correctly.

test_empty_spans: Confirms that an empty list of spans returns an empty result without errors.


### test_names.py

test_redact_person_names_spacy: Tests if SpaCy detects and redacts a person’s name in the text.

test_redact_person_names_hf: Verifies that Hugging Face NER detects and redacts a person’s name in the text.

test_redact_person_names_regex: Tests if regex successfully identifies and redacts multiple person names in the text.


### test_phones.py

test_redact_phone_numbers_spacy: Checks if SpaCy correctly detects and redacts a phone number in the text.

test_redact_phone_numbers_regex: Verifies that regex identifies multiple phone numbers in various formats and redacts them.


### test_process_file.py

test_process_file_success: Tests process_file with mocked functions to verify that redactions are applied correctly, the output file is written, and statistics are updated accurately.

test_process_file_read_error: Verifies that an error message is output to stderr when there is an issue reading the input file.


### test_redact_email_headers.py

test_redact_names_in_headers: Tests redact_email_headers to confirm that names in email headers are correctly detected and redacted, and the statistics are updated accurately.


### test_redact_entities_regex.py

test_no_matches: Checks that no redactions are applied if the text does not contain any of the specified entities.

test_overlapping_matches: Ensures that overlapping entities (e.g., name and phone number) are handled correctly by the regex redaction method, and statistics are updated accurately.


### test_write_stats.py

test_write_stats_to_file: Tests writing redaction statistics to a file by verifying the file content.

test_write_stats_to_stdout: Checks if redaction statistics are correctly output to stdout.

test_write_stats_to_stderr: Ensures redaction statistics are written to stderr when specified.

test_write_stats_to_file_failure: Verifies that an error message is output to stderr if there is a failure in writing statistics to a file.

























