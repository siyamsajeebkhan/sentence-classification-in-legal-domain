import re
from luima_sbd import sbd_utils as luima
import random
import numpy as np
import spacy
import pandas as pd
from spacy.symbols import ORTH
import fasttext
from joblib import load
import sys

random.seed(42)
np.random.seed(42)

nlp = spacy.load("en_core_web_sm")
nlp.tokenizer.add_special_case('Vet. App.', [{ORTH: 'Vet. App.'}])
nlp.tokenizer.add_special_case('Fed. Cir.', [{ORTH: 'Fed. Cir.'}])

def make_span_data(doc, sentences, sentence_offsets):
    span_data=[]
    for sentence, offset in zip(sentences, sentence_offsets):
        start = offset[0]
        end = offset[1]

        document_txt = doc
        sd = {
            'txt': sentence,
            'start': start,
            'start_normalized': start / len(document_txt),
            'end': end
            }
        span_data.append(sd)
    return span_data

def custom_spacy_tokenize(txt):

    nlp.disable_pipes('parser')
    doc = nlp.pipe(txt, n_process=4)
    doc = nlp(txt)
    tokens = list(doc)
    clean_tokens = []
    for i, token in enumerate(tokens):
        if token.pos_ == 'PUNCT' and not re.search("^[0-9]{2}/[0-9]{2}/([0-9]{2}|[0-9]{4})$", token.text):
            pass
        
        elif token.pos_ == 'NUM':
            refined_token = re.sub(r'\W', '', token.text)
            clean_tokens.append(f'<NUM{len(refined_token)}>')
            
        elif token.text == "\'s" and token.pos_ == 'PART':
            pos_token = tokens[i-1].text + token.text
            clean_tokens.pop(len(clean_tokens)-1)
            clean_tokens.append(pos_token.lower())
                   
        elif "-" in token.text:
            splitted_tokens = token.text.split("-")

            for sp_token in splitted_tokens:
                refined_token = re.sub(r'\W', '', sp_token.lower())
                if refined_token != "":
                    if refined_token.isnumeric():
                        refined_token = f'<NUM{len(refined_token)}>'
                    clean_tokens.append(refined_token)
        elif token.text in ("Vet. App.", "Fed. Cir."):
            clean_tokens.append(token.lemma_.lower())
        else:
            refined_token = re.sub(r'\W', '', token.lemma_.lower())
            if re.search('\d+', refined_token) and re.search('[a-zA-Z]+', refined_token):
                continue
            elif refined_token != "" and refined_token.isnumeric():
                refined_token = f'<NUM{len(refined_token)}>'
                clean_tokens.append(refined_token)

            elif refined_token != "":
                clean_tokens.append(refined_token)
                    
    return clean_tokens

def custom_spans_add_spacy_tokens(spans):
    for s in spans:
        s['tokens_spacy'] = custom_spacy_tokenize(s['txt'])
        s['token_count'] = len(s['tokens_spacy'])

def make_word_embedded_feature_vectors_and_labels(spans, model):
    df = pd.DataFrame([s['token_count'] for s in spans])
    df.columns = ['token_count']
    # token_count_mean, token_count_std = df['token_count'].mean(), df['token_count'].std()
    token_count_mean, token_count_std = 21.035180722891567, 15.719815094996603
    final_word_vector = []
    for s in spans:
        if (len(s['tokens_spacy'])):
            word_vector = np.mean(np.array([model.get_word_vector(token) for token in s['tokens_spacy']]), axis=0)
            final_word_vector.append(word_vector)
    starts_normalized = np.array([s['start_normalized'] for s in spans])
    token_count_normalized = np.array([(s['token_count']-token_count_mean)/token_count_std for s in spans])
    X = np.concatenate((np.array(final_word_vector), np.expand_dims(starts_normalized, axis=1), np.expand_dims(token_count_normalized, axis=1)), axis=1)
    return X

def make_tfidf_feature_vectors_and_labels(spans, vectorizer):
    # function takes long to execute
    # note: we un-sparse the matrix here to be able to manipulate it
    
    df = pd.DataFrame([s['token_count'] for s in spans])
    df.columns = ['token_count']
    token_count_mean, token_count_std = df['token_count'].mean(), df['token_count'].std()

    feature_vector = vectorizer.transform([s['txt'] for s in spans]).toarray()
    starts_normalized = np.array([s['start_normalized'] for s in spans])
    token_count_normalized = np.array([(s['token_count']-token_count_mean)/token_count_std for s in spans])

    X = np.concatenate((feature_vector, np.expand_dims(starts_normalized, axis=1), np.expand_dims(token_count_normalized, axis=1)), axis=1)
    return X

def print_classification_report(sentences, labels):
    pred_file = 'predictions.txt'
    file = open(pred_file, 'w')
    for sent, label in zip(sentences, labels):
        to_write = f"{sent} -----> {label}"
        print_pattern = "-"*100
        line_to_write = f"{print_pattern}\n{to_write}\n"
        print(line_to_write)
        file.write(line_to_write)

def classify(file_path):
    """
    Returns a given BVA decision by classifying it

        Parameters:
                    file_path (str): Path to the BVA decision file

        Returns:
                sentences (str): splitted sentences from the BVA decision file using LUIMA segmenter
                labels (str): lables of the splitted sentences
    """

    doc = open(file_path, 'rb').read().decode('latin-1', 'ignore').strip()

    # doc = open(file_path, 'rb').read().decode('latin-1', 'ignore')
    # print(doc)
    sentences = luima.text2sentences(doc, offsets=False)
    sentence_offsets = luima.text2sentences(doc, offsets=True)

    spans = make_span_data(doc, sentences, sentence_offsets)
    custom_spans_add_spacy_tokens(spans)

    # Classification using Word embedding features
    vectorizer = fasttext.load_model("models/ft_word_embedding_model.bin")
    feature_vector = make_word_embedded_feature_vectors_and_labels(spans, vectorizer)
    clf = load('models/word_embedding_best_model_svc_rbf.joblib') 
    labels = clf.predict(feature_vector)

    return sentences, labels
    

if __name__=="__main__":
    file_path = sys.argv[1]
    sentences, labels = classify(file_path)
    print_classification_report(sentences, labels)