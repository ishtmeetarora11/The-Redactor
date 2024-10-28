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
   {"label": "PHONE", "pattern": [{"IS_DIGIT": True, "LENGTH": 3}, {"ORTH": "-"}, {"IS_DIGIT": True, "LENGTH": 3},
                                  {"ORTH": "-"}, {"IS_DIGIT": True, "LENGTH": 4}]},
   # Format: (123) 456-7890
   {"label": "PHONE", "pattern": [{"ORTH": "("}, {"IS_DIGIT": True, "LENGTH": 3}, {"ORTH": ")"},
                                  {"IS_SPACE": True, "OP": "?"}, {"IS_DIGIT": True, "LENGTH": 3}, {"ORTH": "-"},
                                  {"IS_DIGIT": True, "LENGTH": 4}]},
   # format: +1 123-456-7890
   {"label": "PHONE", "pattern": [{"ORTH": "+"}, {"IS_DIGIT": True, "LENGTH": 1}, {"IS_SPACE": True},
                                  {"IS_DIGIT": True, "LENGTH": 3}, {"ORTH": "-"}, {"IS_DIGIT": True, "LENGTH": 3},
                                  {"ORTH": "-"}, {"IS_DIGIT": True, "LENGTH": 4}]},
   # Format: 1-123-456-7890
   {"label": "PHONE", "pattern": [{"IS_DIGIT": True, "LENGTH": 1}, {"ORTH": "-"}, {"IS_DIGIT": True, "LENGTH": 3},
                                  {"ORTH": "-"}, {"IS_DIGIT": True, "LENGTH": 3}, {"ORTH": "-"},
                                  {"IS_DIGIT": True, "LENGTH": 4}]},
   # Format: 123.456.7890
   {"label": "PHONE", "pattern": [{"IS_DIGIT": True, "LENGTH": 3}, {"ORTH": ".", "OP": "?"},
                                  {"IS_DIGIT": True, "LENGTH": 3}, {"ORTH": ".", "OP": "?"},
                                  {"IS_DIGIT": True, "LENGTH": 4}]},
   # Format: 123 456 7890
   {"label": "PHONE", "pattern": [{"IS_DIGIT": True, "LENGTH": 3}, {"IS_SPACE": True}, {"IS_DIGIT": True, "LENGTH": 3},
                                  {"IS_SPACE": True}, {"IS_DIGIT": True, "LENGTH": 4}]},
   # Format: 512) 263-0177
   # Assumption: a starting '(' is missing, making it optional.
   {"label": "PHONE", "pattern": [{"ORTH": "(", "OP": "?"}, {"IS_DIGIT": True, "LENGTH": 3}, {"ORTH": ")"},
                                  {"IS_SPACE": True, "OP": "?"}, {"IS_DIGIT": True, "LENGTH": 3}, {"ORTH": "-"},
                                  {"IS_DIGIT": True, "LENGTH": 4}]},
]


date_patterns = [
   {"label": "DATE", "pattern": [{"SHAPE": "dd"}, {"IS_ALPHA": True}, {"SHAPE": "dddd"}]},
  
   {"label": "DATE", "pattern": [{"SHAPE": "dd/dd/dddd"}]},
  
   {"label": "DATE", "pattern": [{"IS_ALPHA": True}, {"SHAPE": "dd, "}, {"SHAPE": "dddd"}]},
  
   {"label": "DATE", "pattern": [{"LOWER": {"IN": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep",
                                                   "oct", "nov", "dec"]}}, {"ORTH": "/"},
                                 {"IS_DIGIT": True, "LENGTH": 2}, {"ORTH": "/"}, {"IS_DIGIT": True, "LENGTH": 4}]},
  
   {"label": "DATE", "pattern": [{"LOWER": {"IN": ["january", "february", "march", "april", "may", "june", "july",
                                                   "august", "september", "october", "november", "december"]}},
                                 {"IS_DIGIT": True, "LENGTH": 2}, {"ORTH": ","}, {"IS_DIGIT": True, "LENGTH": 4}]},
  
   {"label": "DATE", "pattern": [{"LOWER": {"IN": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sept",
                                                   "oct", "nov", "dec"]}}, {"IS_DIGIT": True, "LENGTH": 2},
                                 {"ORTH": ","}, {"IS_DIGIT": True, "LENGTH": 4}]},
  
   {"label": "DATE", "pattern": [{"IS_ALPHA": True}, {"ORTH": "."}, {"SHAPE": "dd"}]},
  
   {"label": "DATE", "pattern": [{"SHAPE": "dd/dd/dddd"}]},
  
   {"label": "DATE", "pattern": [{"IS_ALPHA": True}, {"SHAPE": "dd"}]},
  
   {"label": "DATE", "pattern": [{"SHAPE": "dd/dd/dd"}]},
  
   {"label": "DATE", "pattern": [{"SHAPE": "dd/dd/dddd"}]},
  
   {"label": "DATE", "pattern": [{"IS_ALPHA": True}, {"SHAPE": "dd,"}, {"SHAPE": "dddd"}]}
  
]


address_patterns = [
  
   {"label": "ADDRESS", "pattern": [
       {"LIKE_NUM": True},
       {"IS_ALPHA": True, "OP": "+"}
   ]},
  
   {"label": "ADDRESS", "pattern": [
       {"ENT_TYPE": "GPE", "OP": "+"},
       {"IS_PUNCT": True, "OP": "?"},
       {"IS_SPACE": True, "OP": "?"},
       {"SHAPE": "XX"}, 
       {"LIKE_NUM": True, "LENGTH": 5} 
   ]},
  
]


filterwarnings('ignore')




nlp = spacy.load('en_core_web_lg')
ruler = nlp.add_pipe("entity_ruler", before="ner")


ruler.add_patterns(phone_patterns)
ruler.add_patterns(date_patterns)
ruler.add_patterns(address_patterns)




hf_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
hf_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
hf_nlp = pipeline(
   "ner", model=hf_model, tokenizer=hf_tokenizer
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


def redact_concept_sentences(doc, concepts):
  
   concept_spans = []
  
   escaped_concepts = [re.escape(concept.lower()) for concept in concepts]
  
   pattern = r'\b(' + '|'.join(escaped_concepts) + r')\b'
  
   for sent in doc.sents:
       if re.search(pattern, sent.text.lower()):
           concept_spans.append((sent.start_char, sent.end_char))
   return concept_spans


def censor_with_spacy(text, entities_to_censor, stats, concepts=None):
  
   doc = nlp(text)
   spans_to_redact = []
  
   spacy_label_to_stat_key = {
       'PERSON': 'names',
       'DATE': 'dates',
       'PHONE': 'phones',
       'GPE': 'addresses',
       'LOC': 'addresses',
       'FAC': 'addresses',
   }
  
   for ent in doc.ents:
       stat_key = spacy_label_to_stat_key.get(ent.label_)
       if stat_key and stat_key in entities_to_censor:
          
           if ent.label_ == 'DATE':
              
               date_tokens = []
               for token in ent:
                   if token.like_num or token.text.lower() in [
                       'january', 'february', 'march', 'april', 'may', 'june', 'july',
                       'august', 'september', 'october', 'november', 'december',
                       'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
                       'sep', 'sept', 'oct', 'nov', 'dec',
                       'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
                       'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'
                   ]:
                       date_tokens.append(token)
               if date_tokens:
                   new_start = date_tokens[0].idx
                   new_end = date_tokens[-1].idx + len(date_tokens[-1])
                   spans_to_redact.append((new_start, new_end))
                   stats[stat_key] += 1
           else:
               spans_to_redact.append((ent.start_char, ent.end_char))
               stats[stat_key] += 1
  
   if concepts:
       concept_spans = redact_concept_sentences(doc, concepts)
       spans_to_redact.extend(concept_spans)
       stats['concepts'] += len(concept_spans)
   return spans_to_redact






def censor_with_hf(text, entities_to_censor, stats):
   ner_results = hf_nlp(text)
   spans_to_redact = []
   # Map Hugging Face entity labels to stats keys
   hf_label_to_stat_key = {
       'PER': 'names',
       'LOC': 'addresses',
       'ORG': 'addresses',  # Include if desired
       'MISC': None,
   }
   # Collect entity spans to redact
   for ent in ner_results:
       label = ent['entity']  # This is the key used when not using aggregation
       stat_key = hf_label_to_stat_key.get(label)
       if stat_key and stat_key in entities_to_censor:
           spans_to_redact.append((ent['start'], ent['end']))
           stats[stat_key] += 1
   return spans_to_redact




def censor_with_regex(text, entities_to_censor, stats):
   """
   Use regex patterns to identify and collect spans to redact.


   Args:
       text (str): The text to process.
       entities_to_censor (list of str): List of entity types to censor.
       stats (dict): Dictionary to track redaction statistics.


   Returns:
       list of tuples: List of (start_char, end_char) tuples for spans to redact.
   """
   spans_to_redact = []
   # Regex pattern for phone numbers
   if 'phones' in entities_to_censor:
       phone_pattern = re.compile(
           r'\b(\+?\d{1,2}\s?)?(\(?\d{3}\)?[\s.-]?|\d{3}[\s.-]?)\d{3}[\s.-]?\d{4}\b'
       )
       for match in phone_pattern.finditer(text):
           spans_to_redact.append((match.start(), match.end()))
           stats['phones'] += 1
   # Regex pattern for dates
   if 'dates' in entities_to_censor:
       date_pattern = re.compile(
           r'\b(?:\d{1,2}[/-])?\d{1,2}[/-]\d{2,4}\b'
       )
       for match in date_pattern.finditer(text):
           spans_to_redact.append((match.start(), match.end()))
           stats['dates'] += 1
   if 'addresses' in entities_to_censor:
       address_pattern = re.compile(
           r'\b\d{1,5}\s+(?:[A-Za-z0-9]+\s){0,5}(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|'
           r'Lane|Ln\.?|Drive|Dr\.?|Court|Ct\.?|Highway|Hwy\.?|Place|Pl\.?|Square|Sq\.?)\.?\b',
           re.IGNORECASE
       )
       for match in address_pattern.finditer(text):
           spans_to_redact.append((match.start(), match.end()))
           stats['addresses'] += 1
   # Additional regex patterns can be added here...
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
       spans_to_redact.extend(censor_with_spacy(text, entities_to_censor, stats, args.concept))
       spans_to_redact.extend(censor_with_hf(text, entities_to_censor, stats))
       spans_to_redact.extend(censor_with_regex(text, entities_to_censor, stats))


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
   