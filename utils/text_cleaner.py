import re

def clean_text(text):

    # remove email headers
    text = re.sub(r'Xref:.*', '', text)
    text = re.sub(r'Path:.*', '', text)

    # remove emails
    text = re.sub(r'\S+@\S+', '', text)

    # remove non letters
    text = re.sub(r'[^a-zA-Z ]', ' ', text)

    # lowercase
    text = text.lower()

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()