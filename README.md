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


## Functions

#### main.py 

The main(url) function runs everything together:

1. **`incident_data = fetchIncidents(url)`**:
   - Fetches incident data from the provided URL.

2. **`incidents = extractIncidents(incident_data)`**:
   - Extracts and processes incidents from the fetched data.

3. **`db = createdb()`**:
   - Initializes the database.

4. **`populatedb(db, incidents)`**:
   - Inserts the extracted incidents into the initialized database.

5. **`status(db)`**:
   - Checks or reports the status of the database.

### fetchincidents.py

```
def fetchIncidents(url, headers={}):

    This function fetches an incident PDF file from a given URL using Python’s urllib.request library. It sends an HTTP request to the specified URL, optionally including additional headers, and returns the content of the PDF as a BytesIO object.

    Args:
        url (str): The URL from which the incident PDF is to be downloaded.
        headers (dict, optional): A dictionary of optional HTTP headers to be included in the request. If no headers are provided, an empty dictionary is used by default.
        
    Returns:
        BytesIO: A BytesIO object containing the binary content of the incident PDF file. Using this later to read and process the PDF data in memory.

```

### extractincidents.py

```
def extractIncidents(incident_data):

    This function processes binary incident data from a PDF and extracts relevant incident records. It uses the PyPDF library to read the PDF and regular expressions to parse the data into structured information.

    Args:
        incident_data (bytes): Binary data of the PDF file that contains incident information, fetched from the URL.
        
    Returns:
        list: list of dictionaries, where each dictionary represents an individual incident. Each dictionary contains the following keys:
        'Date_Time': The date and time of the incident.
        'Incident Number': A unique identifier for the incident.
        'Location': The location where the incident occurred.
        'Nature': The nature or type of the incident.
        'ORI': The originating agency identifier.

```
### db.py

```
def createdb():

    This function is responsible for creating a fresh SQLite database and initializing a table for storing incident records. If the database already exists, it deletes it before creating a new one.
        
    connection: An SQLite database connection object that can be used for further database operations.

    An SQL command is executed to create a table named incidents. The table contains the following fields:

    incident_time (TEXT): The date and time of the incident.
    incident_number (TEXT): The incident’s unique identifier.
    incident_location (TEXT): The location of the incident.
    nature (TEXT): The type or nature of the incident.
    incident_ori (TEXT): The originating agency identifier.

```
```
def populatedb(conn, incidents):

    This function populates the database with incident records. It inserts data into the incidents table using a list of dictionaries, each representing an incident.

    Args:
        conn (sqlite3.Connection): The database connection object created by createdb().

        incidents (list): A list of dictionaries, where each dictionary represents an incident with fields like 'Date_Time', 'Incident Number', 'Location', 'Nature', and 'ORI'.
        
    Inserting data:
        The incidents list is transformed into a list of tuples, each containing values in the following order: Date_Time, Incident Number, Location, Nature, and ORI.

        The executemany method is used to insert multiple rows into the incidents table. Each tuple is inserted into the table using placeholders (?, ?, ?, ?, ?), corresponding to the five columns.

```
```
def status(conn):

    This function queries the database and outputs a summary of incident types (nature) along with the number of occurrences for each type.

    Args:
        conn (sqlite3.Connection): The database connection object created by createdb().
        
    Printing Results:

        An SQL SELECT statement is executed to count the occurrences of each type of incident (nature) in the incidents table. The results are grouped by the nature column and sorted alphabetically.

        The function iterates through the result set, printing each incident type (nature) and its count in the format nature|count.

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

