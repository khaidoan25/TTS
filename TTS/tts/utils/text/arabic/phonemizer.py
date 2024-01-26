import re
from unidecode import unidecode

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

def convert_to_ascii(text):
  return unidecode(text)


def lowercase(text):
  return text.lower()


def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)


def transliteration_cleaners(text):
  '''Pipeline for non-English text that transliterates to ASCII.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def arabic_text_to_phonemes(
    text,
    phoneme_type="ascii"
):
    if phoneme_type == "ascii":
        return transliteration_cleaners(text)
    elif phoneme_type == "buckwalter":
        raise NotImplementedError("Will be updated later")
    elif phoneme_type == "open-source":
        raise NotImplementedError("Will be updated later")
    else:
        raise NotImplementedError("Must be ascii|buckwalter|open-source")
    