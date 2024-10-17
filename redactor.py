import argparse
import glob
import os
import re
import sys
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
from warnings import filterwarnings


filterwarnings('ignore')

nlp = spacy.load('en_core_web_sm')

def redact_names(doc):
    redacted_tokens = []
    for token in doc:
        if token.ent_type_ == 'PERSON':
            redacted_tokens.append('█' * len(token.text))
        else:
            redacted_tokens.append(token.text)
    return " ".join(redacted_tokens)

def redact_dates(doc):
    redacted_tokens = []
    for token in doc:
        if token.ent_type_ in ['DATE', 'TIME']:
            redacted_tokens.append('█' * len(token.text))
        else:
            redacted_tokens.append(token.text)
    return " ".join(redacted_tokens)

def redact_phones(doc):
    pattern = re.compile(r'(\+?\d{1,2}\s?)?(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}')
    redacted_text = pattern.sub(lambda x: '█' * len(x.group()), doc.text)
    return redacted_text

def redact_addresses(doc):
    redacted_tokens = []
    for token in doc:
        if token.ent_type_ in ['GPE', 'LOC']:
            redacted_tokens.append('█' * len(token.text))
        else:
            redacted_tokens.append(token.text)
    return " ".join(redacted_tokens)

def redact_concepts(doc, concepts):
    redacted_text = doc.text
    for concept in concepts:
        redacted_text = re.sub(rf"\b{re.escape(concept)}\b", lambda x: '█' * len(x.group()), redacted_text, flags=re.IGNORECASE)
    return redacted_text

def process_file(file_path, args, stats):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        doc = nlp(text)

        redacted_text = doc.text

        # Redaction steps
        if args.names:
            redacted_text = redact_names(nlp(redacted_text))
            stats['names'] += 1
        if args.dates:
            redacted_text = redact_dates(nlp(redacted_text))
            stats['dates'] += 1
        if args.phones:
            redacted_text = redact_phones(nlp(redacted_text))
            stats['phones'] += 1
        if args.address:
            redacted_text = redact_addresses(nlp(redacted_text))
            stats['addresses'] += 1
        if args.concept:
            redacted_text = redact_concepts(nlp(redacted_text), args.concept)
            stats['concepts'] += 1

        base_name = os.path.basename(file_path)
        censored_file_name = os.path.join(args.output, base_name + '.censored')
        with open(censored_file_name, 'w', encoding='utf-8') as f:
            f.write(redacted_text)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def write_stats(stats, destination):
    stats_text = (
        f"Names redacted: {stats['names']}\n"
        f"Dates redacted: {stats['dates']}\n"
        f"Phone numbers redacted: {stats['phones']}\n"
        f"Addresses redacted: {stats['addresses']}\n"
        f"Concepts redacted: {stats['concepts']}\n"
    )
    if destination == 'stderr':
        sys.stderr.write(stats_text)
    elif destination == 'stdout':
        sys.stdout.write(stats_text)
    else:
        with open(destination, 'w', encoding='utf-8') as f:
            f.write(stats_text)

def main():
    parser = argparse.ArgumentParser(description='Redact sensitive information from text files.')
    parser.add_argument('--input', action='append', required=True, help='Input file glob pattern')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--names', action='store_true', help='Redact names')
    parser.add_argument('--dates', action='store_true', help='Redact dates')
    parser.add_argument('--phones', action='store_true', help='Redact phone numbers')
    parser.add_argument('--address', action='store_true', help='Redact addresses')
    parser.add_argument('--concept', action='append', help='Redact concepts')
    parser.add_argument('--stats', required=True, help='Stats output (stderr, stdout, or filename)')
    args = parser.parse_args()

    stats = {
        'names': 0,
        'dates': 0,
        'phones': 0,
        'addresses': 0,
        'concepts': 0
    }

    os.makedirs(args.output, exist_ok=True)

    for pattern in args.input:
        for file_path in glob.glob(pattern):
            process_file(file_path, args, stats)

    write_stats(stats, args.stats)

if __name__ == '__main__':
    main()
