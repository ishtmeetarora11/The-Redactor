import argparse
import glob
import os
import re
import sys
from warnings import filterwarnings
import spacy
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Define custom patterns for phone numbers
phone_patterns = [
    # Format: 123-456-7890
    {"label": "PHONE", "pattern": [{"SHAPE": "ddd"}, {"ORTH": "-"}, {"SHAPE": "ddd"}, {"ORTH": "-"}, {"SHAPE": "dddd"}]},
    # Format: (123) 456-7890
    {"label": "PHONE", "pattern": [{"ORTH": "("}, {"SHAPE": "ddd"}, {"ORTH": ")"}, {"SHAPE": "ddd"}, {"ORTH": "-"}, {"SHAPE": "dddd"}]},
    # Format: +1 123-456-7890
    {"label": "PHONE", "pattern": [{"ORTH": "+"}, {"SHAPE": "d"}, {"IS_SPACE": True}, {"SHAPE": "ddd"}, {"ORTH": "-"}, {"SHAPE": "ddd"}, {"ORTH": "-"}, {"SHAPE": "dddd"}]},
    # Format: 1-123-456-7890
    {"label": "PHONE", "pattern": [{"SHAPE": "d"}, {"ORTH": "-"}, {"SHAPE": "ddd"}, {"ORTH": "-"}, {"SHAPE": "ddd"}, {"ORTH": "-"}, {"SHAPE": "dddd"}]},
    # Format: 123.456.7890
    {"label": "PHONE", "pattern": [{"SHAPE": "ddd"}, {"ORTH": "."}, {"SHAPE": "ddd"}, {"ORTH": "."}, {"SHAPE": "dddd"}]},
    # Format: 123 456 7890
    {"label": "PHONE", "pattern": [{"SHAPE": "ddd"}, {"IS_SPACE": True}, {"SHAPE": "ddd"}, {"IS_SPACE": True}, {"SHAPE": "dddd"}]},
    # Format: 1234567890
    {"label": "PHONE", "pattern": [{"SHAPE": "dddddddddd"}]},
]

# Define custom patterns for dates
date_patterns = [
    # Format: 14 Jun 2000
    {"label": "DATE", "pattern": [{"SHAPE": "dd"}, {"IS_ALPHA": True}, {"SHAPE": "dddd"}]},
    # Format: Jun 14, 2000
    {"label": "DATE", "pattern": [{"IS_ALPHA": True}, {"SHAPE": "dd,"}, {"SHAPE": "dddd"}]},
    # Format: 06/14/2000
    {"label": "DATE", "pattern": [{"SHAPE": "dd/dd/dddd"}]},
    # Format: 06-14-2000
    {"label": "DATE", "pattern": [{"SHAPE": "dd-dd-dddd"}]},
    # Format: 14/06/2000
    {"label": "DATE", "pattern": [{"SHAPE": "dd/dd/dddd"}]},
    # Format: 14-06-2000
    {"label": "DATE", "pattern": [{"SHAPE": "dd-dd-dddd"}]},
    # Format: June 14, 2000
    {"label": "DATE", "pattern": [{"IS_ALPHA": True}, {"SHAPE": "dd,"}, {"SHAPE": "dddd"}]},
]

# Define custom patterns for addresses
address_patterns = [
    # Format: 123 Main Street
    {"label": "ADDRESS", "pattern": [{"LIKE_NUM": True}, {"IS_ALPHA": True, "OP": "+"}, {"LOWER": {"IN": ["street", "st", "avenue", "ave", "road", "rd", "boulevard", "blvd", "lane", "ln", "drive", "dr", "court", "ct", "highway", "hwy", "place", "pl", "square", "sq", "building", "bldg", "apartment", "apt", "suite", "ste"]}}]},
    # Format: 123 Main St.
    {"label": "ADDRESS", "pattern": [{"LIKE_NUM": True}, {"IS_ALPHA": True, "OP": "+"}, {"LOWER": {"IN": ["street", "st.", "avenue", "ave.", "road", "rd.", "boulevard", "blvd.", "lane", "ln.", "drive", "dr.", "court", "ct.", "highway", "hwy.", "place", "pl.", "square", "sq.", "building", "bldg.", "apartment", "apt.", "suite", "ste."]}}]},
]

# Add custom patterns for names in email addresses
name_patterns = [
    # Matches email addresses like 'robert.badeer@enron.com'
    {"label": "PERSON", "pattern": [{"LOWER": {"REGEX": "^[a-z]+(\\.[a-z]+)+$"}}]},
    # Matches names with underscores like 'robert_badeer'
    {"label": "PERSON", "pattern": [{"LOWER": {"REGEX": "^[a-z]+(_[a-z]+)+$"}}]},
    # Matches capitalized names in headers and signatures
    {"label": "PERSON", "pattern": [{"IS_TITLE": True}, {"IS_TITLE": True, "OP": "+"}]},
]

filterwarnings('ignore')

nlp = spacy.load('en_core_web_lg')
ruler = nlp.add_pipe("entity_ruler", before="ner")
ruler.add_patterns(phone_patterns)
ruler.add_patterns(date_patterns)
ruler.add_patterns(address_patterns)
ruler.add_patterns(name_patterns)

# Add sentencizer to handle sentence segmentation in emails
nlp.add_pipe('sentencizer')

hf_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
hf_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
hf_nlp = pipeline(
    "ner", model=hf_model, tokenizer=hf_tokenizer, aggregation_strategy="simple"
)

def merge_spans(spans):
    if not spans:
        return []
    sorted_spans = sorted(spans, key=lambda x: x[0])
    merged = [sorted_spans[0]]
    for current in sorted_spans[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    return merged

def redact_concept_sentences(text, concepts):
    concept_spans = []
    escaped_concepts = [re.escape(concept.lower()) for concept in concepts]
    pattern = re.compile(r'\b(' + '|'.join(escaped_concepts) + r')\b', re.IGNORECASE)

    # Split the text into sentences using a combination of punctuation and newlines
    sentences = []
    sentence_endings = re.finditer(r'.+?(?:[\.\!\?](?=\s)|\n|$)', text, re.DOTALL)
    for match in sentence_endings:
        sentence = match.group()
        start_char = match.start()
        end_char = match.end()
        sentences.append((sentence, start_char, end_char))

    for sentence, start_char, end_char in sentences:
        if pattern.search(sentence):
            concept_spans.append((start_char, end_char))

    return concept_spans

def censor_with_spacy(text, entities_to_censor, stats):
    doc = nlp(text)
    spans_to_redact = []

    spacy_label_to_stat_key = {
        'PERSON': 'names',
        'DATE': 'dates',
        'TIME': 'dates',
        'PHONE': 'phones',
        # Removed 'GPE', 'LOC', 'FAC' to prevent over-redaction
    }

    for ent in doc.ents:
        stat_key = spacy_label_to_stat_key.get(ent.label_)
        if stat_key and stat_key in entities_to_censor:
            spans_to_redact.append((ent.start_char, ent.end_char))
            stats[stat_key] += 1

    return spans_to_redact

def censor_with_hf(text, entities_to_censor, stats):
    ner_results = hf_nlp(text)
    spans_to_redact = []
    # Map Hugging Face entity labels to stats keys
    hf_label_to_stat_key = {
        'PER': 'names',
        'LOC': 'addresses',
        # 'ORG': 'names',  # Removed to prevent organization names from being redacted
        'MISC': None,
    }
    # Collect entity spans to redact
    for ent in ner_results:
        label = ent['entity_group']
        stat_key = hf_label_to_stat_key.get(label)
        if stat_key and stat_key in entities_to_censor:
            spans_to_redact.append((ent['start'], ent['end']))
            stats[stat_key] += 1
    return spans_to_redact

def censor_email_headers(text, entities_to_censor, stats):
    spans_to_redact = []
    if 'names' in entities_to_censor:
        # Regex pattern to match email headers with potential names
        header_pattern = re.compile(
            r'^(From|To|Cc|Bcc|X-From|X-To|X-cc|X-bcc):\s*(.*)',
            re.IGNORECASE | re.MULTILINE
        )
        for match in header_pattern.finditer(text):
            header_content = match.group(2)
            # Extract names using regex
            name_matches = re.finditer(
                r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', header_content)
            for name_match in name_matches:
                start = match.start(2) + name_match.start()
                end = match.start(2) + name_match.end()
                spans_to_redact.append((start, end))
                stats['names'] += 1

            # Extract email addresses and redact names within them
            email_matches = re.finditer(
                r'\b([\w\.-]+)@([\w\.-]+\.\w+)\b', header_content)
            for email_match in email_matches:
                local_part = email_match.group(1)
                name_parts = re.split(r'[._]', local_part)
                current_pos = match.start(2) + email_match.start(1)
                for part in name_parts:
                    if part.isalpha():
                        start = current_pos
                        end = start + len(part)
                        spans_to_redact.append((start, end))
                        stats['names'] += 1
                    current_pos += len(part) + 1  # +1 for the separator
    return spans_to_redact

def censor_with_regex(text, entities_to_censor, stats):
    spans_to_redact = []
    # Regex pattern for names
    if 'names' in entities_to_censor:
        # Match capitalized names (e.g., 'John Doe')
        name_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b')
        for match in name_pattern.finditer(text):
            spans_to_redact.append((match.start(), match.end()))
            stats['names'] += 1

        # Match names in email addresses (e.g., 'robert.badeer' in 'robert.badeer@enron.com')
        email_name_pattern = re.compile(
            r'\b([a-z]+(?:[\._][a-z]+)+)@[\w\.-]+\b', re.IGNORECASE)
        for match in email_name_pattern.finditer(text):
            local_part = match.group(1)
            name_parts = re.split(r'[._]', local_part)
            current_pos = match.start(1)
            for part in name_parts:
                if part.isalpha():
                    start = current_pos
                    end = start + len(part)
                    spans_to_redact.append((start, end))
                    stats['names'] += 1
                current_pos += len(part) + 1  # +1 for the separator

    # Regex pattern for phone numbers
    if 'phones' in entities_to_censor:
        phone_pattern = re.compile(
            r'\b(\+?\d{1,2}[\s-])?(\(?\d{3}\)?[\s.-]?|\d{3}[\s.-]?)[\s.-]?\d{3}[\s.-]?\d{4}\b'
        )
        for match in phone_pattern.finditer(text):
            spans_to_redact.append((match.start(), match.end()))
            stats['phones'] += 1

    # Regex pattern for dates
    if 'dates' in entities_to_censor:
        date_pattern = re.compile(
            r'\b(?:\d{1,2}[/-])?\d{1,2}[/-]\d{2,4}\b'
            r'|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|'
            r'Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s\d{1,2},?\s\d{4}\b',
            re.IGNORECASE
        )
        for match in date_pattern.finditer(text):
            spans_to_redact.append((match.start(), match.end()))
            stats['dates'] += 1

    # Regex pattern for addresses
    if 'addresses' in entities_to_censor:
        address_pattern = re.compile(
            r'\b\d{1,5}\s+[\w\s]{1,50}\b(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|'
            r'Lane|Ln\.?|Drive|Dr\.?|Court|Ct\.?|Highway|Hwy\.?|Place|Pl\.?|Square|Sq\.?|Building|Bldg\.?|'
            r'Apartment|Apt\.?|Suite|Ste\.?)\b',
            re.IGNORECASE
        )
        for match in address_pattern.finditer(text):
            spans_to_redact.append((match.start(), match.end()))
            stats['addresses'] += 1

    return spans_to_redact

def write_stats(stats, destination):
    stats_text = (
        f"Names redacted: {stats.get('names', 0)}\n"
        f"Dates redacted: {stats.get('dates', 0)}\n"
        f"Phone numbers redacted: {stats.get('phones', 0)}\n"
        f"Addresses redacted: {stats.get('addresses', 0)}\n"
        f"Concepts redacted: {stats.get('concepts', 0)}\n"
    )
    if destination == 'stderr':
        sys.stderr.write(stats_text)
    elif destination == 'stdout':
        sys.stdout.write(stats_text)
    else:
        with open(destination, 'w', encoding='utf-8') as f:
            f.write(stats_text)

def process_file(file_path, args, stats):
    try:
        # Read the input file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Determine which entities to censor based on command-line arguments
        entities_to_censor = []
        if args.names:
            entities_to_censor.append('names')
        if args.dates:
            entities_to_censor.append('dates')
        if args.phones:
            entities_to_censor.append('phones')
        if args.address:
            entities_to_censor.append('addresses')

        # Collect spans to redact from different models
        spans_to_redact = []
        spans_to_redact.extend(censor_email_headers(text, entities_to_censor, stats))
        spans_to_redact.extend(censor_with_spacy(text, entities_to_censor, stats))
        spans_to_redact.extend(censor_with_hf(text, entities_to_censor, stats))
        spans_to_redact.extend(censor_with_regex(text, entities_to_censor, stats))

        # Handle concept redaction separately
        if args.concept:
            concept_spans = redact_concept_sentences(text, args.concept)
            spans_to_redact.extend(concept_spans)
            stats['concepts'] += len(concept_spans)

        # Merge overlapping spans
        spans_to_redact = merge_spans(spans_to_redact)

        # Apply redactions to the text while preserving formatting
        redacted_text = list(text)
        for start_char, end_char in spans_to_redact:
            for i in range(start_char, end_char):
                # Optionally skip redacting certain characters (e.g., newlines)
                if redacted_text[i] != '\n':
                    redacted_text[i] = '█'
        redacted_text = ''.join(redacted_text)

        # Write the redacted text to the output file
        base_name = os.path.basename(file_path)
        censored_file_name = os.path.join(args.output, f"{base_name}.censored")
        with open(censored_file_name, 'w', encoding='utf-8') as f:
            f.write(redacted_text)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description='Redact sensitive information from text files.'
    )
    parser.add_argument('--input', action='append', required=True, help='Input file glob pattern')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--names', action='store_true', help='Redact names')
    parser.add_argument('--dates', action='store_true', help='Redact dates')
    parser.add_argument('--phones', action='store_true', help='Redact phone numbers')
    parser.add_argument('--address', action='store_true', help='Redact addresses')
    parser.add_argument('--concept', action='append', help='Redact sentences containing specific concepts')
    parser.add_argument('--stats', required=True, help='Stats output destination (stderr, stdout, or filename)')
    args = parser.parse_args()

    # Initialize the statistics dictionary
    stats = {
        'names': 0,
        'dates': 0,
        'phones': 0,
        'addresses': 0,
        'concepts': 0,
    }

    # Create the output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Process each input file matching the glob patterns
    for pattern in args.input:
        for file_path in glob.glob(pattern):
            process_file(file_path, args, stats)

    # Write the redaction statistics to the specified destination
    write_stats(stats, args.stats)

if __name__ == '__main__':
    main()
