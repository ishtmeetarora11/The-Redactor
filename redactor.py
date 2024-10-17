import argparse
import glob
import os
import re
import sys
import spacy
import phonenumbers
from spacy.tokens import Span
from warnings import filterwarnings


filterwarnings('ignore')

nlp = spacy.load('en_core_web_lg')
def collect_phone_spans(text):
    phone_spans = []
    for match in phonenumbers.PhoneNumberMatcher(text, None):  # 'None' detects numbers from all regions
        phone_spans.append((match.start, match.end))
    return phone_spans

def merge_spans(spans):
    if not spans:
        return []
    # Sort spans by start position
    sorted_spans = sorted(spans, key=lambda x: x[0])
    merged = [sorted_spans[0]]
    for current in sorted_spans[1:]:
        last = merged[-1]
        if current[0] <= last[1]:  # Overlapping spans
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    return merged

def process_file(file_path, args, stats):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        doc = nlp(text)
        # for ent in doc.ents:
        #     print(ent.text, ent.label_)
        spans_to_redact = []

        if args.names:
            name_spans = [(ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ == 'PERSON']
            spans_to_redact.extend(name_spans)
            stats['names'] += len(name_spans)
        if args.dates:
            date_spans = [(ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ in ['DATE', 'TIME']]
            spans_to_redact.extend(date_spans)
            stats['dates'] += len(date_spans)
        if args.address:
            address_spans = [(ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ in ['GPE', 'LOC']]
            spans_to_redact.extend(address_spans)
            stats['addresses'] += len(address_spans)
        if args.concept:
            concept_spans = []
            for concept in args.concept:
                for match in re.finditer(rf'\b{re.escape(concept)}\b', text, re.IGNORECASE):
                    concept_spans.append((match.start(), match.end()))
            spans_to_redact.extend(concept_spans)
            stats['concepts'] += len(concept_spans)
        if args.phones:
            phone_spans = collect_phone_spans(text)
            spans_to_redact.extend(phone_spans)
            stats['phones'] += len(phone_spans)

        # Merge overlapping spans
        spans_to_redact = merge_spans(spans_to_redact)

        # Build redacted text
        redacted_text = ''
        last_idx = 0
        for start_char, end_char in spans_to_redact:
            redacted_text += text[last_idx:start_char]
            redacted_text += '█' * (end_char - start_char)
            last_idx = end_char
        redacted_text += text[last_idx:]

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
