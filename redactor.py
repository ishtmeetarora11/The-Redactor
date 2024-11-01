import argparse
import glob
import os
import re
import sys
from warnings import filterwarnings

import spacy
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Suppress all warnings
filterwarnings('ignore')

# Define custom token patterns for phone numbers
PHONE_PATTERNS = [
    # Different phone number formats (e.g., 123-456-7890, (123) 456-7890)
    {"label": "PHONE", "pattern": [{"SHAPE": "ddd"}, {"ORTH": "-"}, {"SHAPE": "ddd"}, {"ORTH": "-"}, {"SHAPE": "dddd"}]},
    {"label": "PHONE", "pattern": [{"ORTH": "("}, {"SHAPE": "ddd"}, {"ORTH": ")"}, {"SHAPE": "ddd"}, {"ORTH": "-"}, {"SHAPE": "dddd"}]},
    {"label": "PHONE", "pattern": [{"ORTH": "+"}, {"SHAPE": "d"}, {"IS_SPACE": True}, {"SHAPE": "ddd"}, {"ORTH": "-"}, {"SHAPE": "ddd"}, {"ORTH": "-"}, {"SHAPE": "dddd"}]},
    {"label": "PHONE", "pattern": [{"SHAPE": "d"}, {"ORTH": "-"}, {"SHAPE": "ddd"}, {"ORTH": "-"}, {"SHAPE": "ddd"}, {"ORTH": "-"}, {"SHAPE": "dddd"}]},
    {"label": "PHONE", "pattern": [{"SHAPE": "ddd"}, {"ORTH": "."}, {"SHAPE": "ddd"}, {"ORTH": "."}, {"SHAPE": "dddd"}]},
    {"label": "PHONE", "pattern": [{"SHAPE": "ddd"}, {"IS_SPACE": True}, {"SHAPE": "ddd"}, {"IS_SPACE": True}, {"SHAPE": "dddd"}]},
    {"label": "PHONE", "pattern": [{"SHAPE": "dddddddddd"}]},
]

# Define custom token patterns for dates
DATE_PATTERNS = [
    # Different date formats (e.g., 14 Jun 2000, 06/14/2000)
    {"label": "DATE", "pattern": [{"SHAPE": "dd"}, {"IS_ALPHA": True}, {"SHAPE": "dddd"}]},
    {"label": "DATE", "pattern": [{"IS_ALPHA": True}, {"SHAPE": "dd,"}, {"SHAPE": "dddd"}]},
    {"label": "DATE", "pattern": [{"SHAPE": "dd/dd/dddd"}]},
    {"label": "DATE", "pattern": [{"SHAPE": "dd-dd-dddd"}]},
    {"label": "DATE", "pattern": [{"IS_ALPHA": True}, {"SHAPE": "dd,"}, {"SHAPE": "dddd"}]},
]

# Define custom token patterns for addresses
ADDRESS_PATTERNS = [
    # Common address formats (e.g., 123 Main Street)
    {"label": "ADDRESS", "pattern": [
        {"LIKE_NUM": True},
        {"IS_ALPHA": True, "OP": "+"},
        {"LOWER": {"IN": [
            "street", "st", "avenue", "ave", "road", "rd", "boulevard", "blvd",
            "lane", "ln", "drive", "dr", "court", "ct", "highway", "hwy",
            "place", "pl", "square", "sq", "building", "bldg", "apartment",
            "apt", "suite", "ste"
        ]}}
    ]},
]

# Define custom token patterns for person names in email addresses
NAME_PATTERNS = [
    # Different name formats in emails or capitalized text
    {"label": "PERSON", "pattern": [{"LOWER": {"REGEX": "^[a-z]+(\\.[a-z]+)+$"}}]},
    {"label": "PERSON", "pattern": [{"LOWER": {"REGEX": "^[a-z]+(_[a-z]+)+$"}}]},
    {"label": "PERSON", "pattern": [{"IS_TITLE": True}, {"IS_TITLE": True, "OP": "+"}]},
]

def initialize_spacy_nlp():
    """
    Initialize and return the SpaCy NLP pipeline with custom patterns for redaction.
    Loads the pipeline only once (lazy loading) for efficiency.
    """
    if not hasattr(initialize_spacy_nlp, "nlp"):
        try:
            nlp = spacy.load('en_core_web_lg')
            # Add entity ruler for custom patterns before the named entity recognizer (NER)
            entity_ruler = nlp.add_pipe("entity_ruler", before="ner")
            entity_ruler.add_patterns(PHONE_PATTERNS + DATE_PATTERNS + ADDRESS_PATTERNS + NAME_PATTERNS)
            nlp.add_pipe('sentencizer')  # Adds sentence segmentation
            initialize_spacy_nlp.nlp = nlp
        except OSError as e:
            sys.stderr.write(
                "SpaCy model 'en_core_web_lg' not found. Install it with: python -m spacy download en_core_web_lg\n"
            )
            raise e
    return initialize_spacy_nlp.nlp

def initialize_hf_pipeline():
    """
    Initialize and return the Hugging Face NER pipeline for redaction.
    Loads the pipeline only once for efficiency.
    """
    if not hasattr(initialize_hf_pipeline, "pipeline"):
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        initialize_hf_pipeline.pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return initialize_hf_pipeline.pipeline

def merge_overlapping_spans(spans):
    """
    Merge overlapping or adjacent character spans.
    """
    if not spans:
        return []

    sorted_spans = sorted(spans, key=lambda x: x[0])
    merged = [sorted_spans[0]]

    for current_start, current_end in sorted_spans[1:]:
        last_start, last_end = merged[-1]
        if current_start <= last_end:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))

    return merged

def identify_concept_sentences(text, concepts):
    """
    Identify sentences that contain specified concepts to redact.
    """
    escaped_concepts = [re.escape(concept.lower()) for concept in concepts]
    concept_pattern = re.compile(r'\b(' + '|'.join(escaped_concepts) + r')\b', re.IGNORECASE)

    sentence_pattern = re.compile(r'.+?(?:[.!?](?=\s)|\n|$)', re.DOTALL)
    concept_spans = []

    for match in sentence_pattern.finditer(text):
        sentence = match.group()
        if concept_pattern.search(sentence):
            concept_spans.append((match.start(), match.end()))

    return concept_spans

def redact_entities_spacy(text, targets, stats):
    """
    Redact entities identified by SpaCy based on specified categories.
    """
    nlp = initialize_spacy_nlp()
    doc = nlp(text)
    redaction_spans = []

    # Mapping entity labels to redaction categories
    label_mapping = {
        'PERSON': 'names',
        'DATE': 'dates',
        'PHONE': 'phones',
        'GPE': 'addresses',
        'LOC': 'addresses'
    }

    for ent in doc.ents:
        category = label_mapping.get(ent.label_)
        if category and category in targets:
            redaction_spans.append((ent.start_char, ent.end_char))
            stats[category] += 1

    return redaction_spans

def redact_entities_hf(text, targets, stats):
    """
    Redact entities identified by Hugging Face NER based on target categories.
    """
    ner_pipeline = initialize_hf_pipeline()
    ner_results = ner_pipeline(text)
    redaction_spans = []

    hf_label_mapping = {
        'PER': 'names',
        'LOC': 'addresses'
    }

    for entity in ner_results:
        category = hf_label_mapping.get(entity['entity_group'])
        if category and category in targets:
            redaction_spans.append((entity['start'], entity['end']))
            stats[category] += 1

    return redaction_spans

def redact_email_headers(text, targets, stats):
    """
    Redact names found in email headers.
    """
    redaction_spans = []

    if 'names' not in targets:
        return redaction_spans

    header_pattern = re.compile(
        r'^(From|To|Cc|Bcc|X-From|X-To|X-cc|X-bcc):\s*(.*)',
        re.IGNORECASE | re.MULTILINE
    )

    for match in header_pattern.finditer(text):
        header_content = match.group(2)

        name_matches = re.finditer(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', header_content)
        for name_match in name_matches:
            start = match.start(2) + name_match.start()
            end = match.start(2) + name_match.end()
            redaction_spans.append((start, end))
            stats['names'] += 1

        email_pattern = re.compile(r'\b([\w\.-]+)@([\w\.-]+\.\w+)\b', re.IGNORECASE)
        for email_match in email_pattern.finditer(header_content):
            local_part = email_match.group(1)
            name_parts = re.split(r'[._]', local_part)
            current_pos = match.start(2) + email_match.start(1)

            for part in name_parts:
                if part.isalpha():
                    start = current_pos
                    end = start + len(part)
                    redaction_spans.append((start, end))
                    stats['names'] += 1
                current_pos += len(part) + 1

    return redaction_spans

def redact_entities_regex(text, targets, stats):
    """
    Redact entities identified by regular expressions based on target categories.
    """
    redaction_spans = []

    if 'names' in targets:
        name_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b')
        for match in name_pattern.finditer(text):
            redaction_spans.append((match.start(), match.end()))
            stats['names'] += 1

        email_name_pattern = re.compile(r'\b([a-z]+(?:[\._][a-z]+)+)@[\w\.-]+\b', re.IGNORECASE)
        for match in email_name_pattern.finditer(text):
            local_part = match.group(1)
            name_parts = re.split(r'[._]', local_part)
            current_pos = match.start(1)
            for part in name_parts:
                if part.isalpha():
                    start = current_pos
                    end = start + len(part)
                    redaction_spans.append((start, end))
                    stats['names'] += 1
                current_pos += len(part) + 1

    if 'phones' in targets:
        phone_pattern = re.compile(r'\b(\+?\d{1,2}[\s-])?(\(?\d{3}\)?[\s.-]?|\d{3}[\s.-]?)[\s.-]?\d{3}[\s.-]?\d{4}\b')
        for match in phone_pattern.finditer(text):
            redaction_spans.append((match.start(), match.end()))
            stats['phones'] += 1

    if 'dates' in targets:
        date_pattern = re.compile(
            r'\b(?:\d{1,2}[/-])?\d{1,2}[/-]\d{2,4}\b|'
            r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|'
            r'Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s\d{1,2},?\s\d{4}\b',
            re.IGNORECASE
        )
        for match in date_pattern.finditer(text):
            redaction_spans.append((match.start(), match.end()))
            stats['dates'] += 1

    if 'addresses' in targets:
        address_pattern = re.compile(
            r'('
            r'\b\d{1,5}\s+(?:[A-Z][a-zA-Z]*(?:\s|$)){1,5}'
            r'(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|' 
            r'Lane|Ln\.?|Drive|Dr\.?|Court|Ct\.?|Highway|Hwy\.?|Place|Pl\.?|' 
            r'Square|Sq\.?|Building|Bldg\.?|Apartment|Apt\.?|Suite|Ste\.?)?'
            r')',
            re.IGNORECASE | re.VERBOSE
        )
        for match in address_pattern.finditer(text):
            redaction_spans.append((match.start(), match.end()))
            stats['addresses'] += 1

    return redaction_spans

def write_stats(stats, destination):
    """
    Output redaction statistics to the specified destination (stderr, stdout, or file).
    """
    stats_report = (
        f"Names redacted: {stats.get('names', 0)}\n"
        f"Dates redacted: {stats.get('dates', 0)}\n"
        f"Phone numbers redacted: {stats.get('phones', 0)}\n"
        f"Addresses redacted: {stats.get('addresses', 0)}\n"
        f"Concepts redacted: {stats.get('concepts', 0)}\n"
    )
    if destination.lower() == 'stderr':
        sys.stderr.write(stats_report)
    elif destination.lower() == 'stdout':
        sys.stdout.write(stats_report)
    else:
        try:
            with open(destination, 'w', encoding='utf-8') as f:
                f.write(stats_report)
        except Exception as e:
            sys.stderr.write(f"Failed to write statistics to {destination}: {e}\n")

def process_file(file_path, args, stats):
    """
    Process and redact a single text file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        sys.stderr.write(f"Error reading file {file_path}: {e}\n")
        return

    entities_to_censor = []
    if args.names:
        entities_to_censor.append('names')
    if args.dates:
        entities_to_censor.append('dates')
    if args.phones:
        entities_to_censor.append('phones')
    if args.address:
        entities_to_censor.append('addresses')

    spans_to_redact = []
    spans_to_redact.extend(redact_email_headers(text, entities_to_censor, stats))
    spans_to_redact.extend(redact_entities_spacy(text, entities_to_censor, stats))
    spans_to_redact.extend(redact_entities_hf(text, entities_to_censor, stats))
    spans_to_redact.extend(redact_entities_regex(text, entities_to_censor, stats))

    if args.concept:
        concept_spans = identify_concept_sentences(text, args.concept)
        spans_to_redact.extend(concept_spans)
        stats['concepts'] += len(concept_spans)

    merged_spans = merge_overlapping_spans(spans_to_redact)

    redacted_text = list(text)
    for start_char, end_char in merged_spans:
        for i in range(start_char, end_char):
            if redacted_text[i] != '\n':
                redacted_text[i] = '█'
    final_text = ''.join(redacted_text)

    base_name = os.path.basename(file_path)
    censored_file_name = os.path.join(args.output, f"{base_name}.censored")

    try:
        with open(censored_file_name, 'w', encoding='utf-8') as f:
            f.write(final_text)
    except Exception as e:
        sys.stderr.write(f"Error writing to file {censored_file_name}: {e}\n")

def main():
    """
    Main function to parse arguments and initiate the redaction process.
    """
    parser = argparse.ArgumentParser(
        description='Redact sensitive information from text files.'
    )
    parser.add_argument('--input', action='append', required=True, help='Input file glob pattern(s)')
    parser.add_argument('--output', required=True, help='Directory to save redacted files')
    parser.add_argument('--names', action='store_true', help='Enable redaction of names')
    parser.add_argument('--dates', action='store_true', help='Enable redaction of dates')
    parser.add_argument('--phones', action='store_true', help='Enable redaction of phone numbers')
    parser.add_argument('--address', action='store_true', help='Enable redaction of addresses')
    parser.add_argument('--concept', action='append', help='Redact sentences containing specified concepts')
    parser.add_argument('--stats', required=True, help='Destination for statistics (stderr, stdout, or filepath)')
    args = parser.parse_args()

    redaction_stats = {
        'names': 0,
        'dates': 0,
        'phones': 0,
        'addresses': 0,
        'concepts': 0,
    }

    os.makedirs(args.output, exist_ok=True)

    for pattern in args.input:
        matched_files = glob.glob(pattern)
        if not matched_files:
            sys.stderr.write(f"No files matched the pattern: {pattern}\n")
        for file_path in matched_files:
            process_file(file_path, args, redaction_stats)

    write_stats(redaction_stats, args.stats)

if __name__ == '__main__':
    main()
