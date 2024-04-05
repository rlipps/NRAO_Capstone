# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Overview
# 
# This notebook converts the proposal text (as stored in a dataiku dataset) into a bag-of-words.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Load libraries

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# If importing en_core_web_sm files, try spacy.cli.download("en_core_web_sm")
import spacy
# spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")


import usbritish
import dataiku
import re
import pandas as pd
import io

from tqdm.notebook import tqdm
MIN_SIZE = 2
import datetime as dt
import header

# Set parallelization
# no more than 8 or 10 workers, otherwise it is too expensive
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, use_memory_fs=False, nb_workers=8)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Set stop-words, phrases, and acronyms

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# If words contain these symbols and numbers they will be removed
numbers_and_symbols = ['0','1','2','3','4','5','6','7','8','9',"=",'<','>','~',"/",'`',".",","]

# Add personalized words to the default stopwords list
nlp.Defaults.stop_words |= {'observation','alma','resolution','source','show','sample','high','use','observe','figure','low','image','propose', 'also','use', 'large', 'study','reference', 'detect','see','well', 'time', 'however', 'expect','provide','datum', 'model', 'result','sensitivity','scale','find','allow','scientific','target','compare','resolve','first', 'leave', 'estimate','suggest', 'due', 'obtain', 'small', 'measure' ,'include','property','justification', 'right ', 'understand', 'similar', 'detection', 'require', 'indicate', 'order', 'range', 'make','map','thus','follow','fig','goal','proposal','field','determine','therefore', 'reveal','give', 'process','total', 'important', 'know','constrain', 'ratio','even','case','et', 'al', 'pc','kpc','apj','km','mm','m','one','two', 'data', 'us','mnras', 'left', 'right', 'may', 'within','would','need','request','mjy','different','assume','recent','good','since','still','previous','science','ghz','could','object','much','survey','three','whether','likely','several','like','able','identify','new','best','number','analysis','confirm','predict','le','evidence','select','example','take','recently','combine','exist','value','fit','objective','comparison','investigate','respectively','many','although','achieve','cm','jy','need','enough','search','yr','explain','au','apjl','per','arxiv'}

# Additional stop words
stop_words_other = {'a&a', 'aa', 'apj', 'apjl', 'mnras', 'pasp', 'aj', 'cycle', 'band',
                  'emission', 'free', 'anticipate', 'originate', 'success', 'separate', 'uv', 'significance',
                  'hot', 'frequency', 'wavelength', 'realistic', 'mas', 'mg', 'minute', 'ii', 'ad', 'hd',
                  'occurrence', 'event', 'myr', 'ra', 'dec',  'ly', 'tau', 'cn',
                  'arc', 'ori', 'hh', 'iii', 'cha', 'ab', 'tw', 'ms', 'ngc', 'pds'}
stop_words_other.update({'jwst','hcn','hco+','oh','xray','aca', 'vla', 'gbt', 'proto'})
stop_words_other.update({'noema','quiescent','nir', 'heating', 'sb','temperature','cr','hya','liu','warm','nh','extent','spitzer', 'co','yang'})
nlp.Defaults.stop_words |= stop_words_other

# Get dataframes that contains  other ways of spelling words and acronyms
acronyms_df = dataiku.Dataset(header.NAME_ACRONYMS).get_dataframe()
phrases_df = dataiku.Dataset(header.NAME_PHRASES).get_dataframe()

# Get list of plural acronyms that should be checked
mask = (acronyms_df['check_plural'] == True)
acronyms_plural = dict()
for index in acronyms_df[mask].index:
    key = acronyms_df['joined'][index]
    acronyms_plural['%ss' % key] = key

# Create dictionaries with the datasets
# phrases = dict(zip(phrases_df.initial, phrases_df.final))
acronyms = dict(zip(acronyms_df.initial, acronyms_df.joined))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
phrases_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Functions to modify text for stopwords, phrases, and acronyms

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def multiple_replace(string, rep_dict):
    """ Replaces several words in a string with other words """
    pattern = re.compile( "|".join([re.escape(k) for k in rep_dict]), flags=re.DOTALL)
    return pattern.sub(lambda x: rep_dict[x.group(0)], string)


def change_text(text):
    """
         Takes a string and performs the following functions:
         1) Replaces dashes and newline characters with spaces
         2) Replaces British spelling of words with their American spelling
         3) Replaces multiple spaces with a single space
         4) Standardize common phrases
         5) Change common phrases to acronyms
    """
    # Change spaces and newline characters
    snew = text.replace('-', ' ').replace('\n',' ')

    # Change words that are case sensitive
    # Replace common astronomy words/phrases with a standard spelling/phrasing
    # These need to be done in stages based on the priority order in the dataframe
    priorities = phrases_df['priority'].unique()
    priorities.sort()
    for priority in priorities:
        mask = (phrases_df['priority'] == priority) & (phrases_df['case_sensitive'] == True)
        if mask.sum() > 0:
            phrases = dict(zip(phrases_df[mask].initial, phrases_df[mask].final))
            snew = multiple_replace(snew, phrases)

    # Convert to lower case and replace () and [] with spaces
    snew = snew.lower().replace('(',' ').replace(')',' ').replace('[',' ').replace(']',' ')

    # Change british spelling to american spelling
    snew = usbritish.change2us(snew)

    # Remove all multiple spaces by splitting the word and rejoining it.
    snew = ' '.join(snew.split())

    # Replace common astronomy words/phrases with a standard spelling/phrasing
    # These need to be done in stages based on the priority order in the dataframe
    priorities = phrases_df['priority'].unique()
    priorities.sort()
    for priority in priorities:
        mask = (phrases_df['priority'] == priority) & (phrases_df['case_sensitive'] == False)
        if mask.sum() > 0:
            phrases = dict(zip(phrases_df[mask].initial, phrases_df[mask].final))
            snew = multiple_replace(snew, phrases)

    # Change common phrases to acronoyms
    final = multiple_replace(snew, acronyms)

    # Return text
    return final

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Read proposal text into dataframe

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
proposal_as_text = dataiku.Dataset(header.NAME_PROPOSAL_AS_TXT)
proposal_as_text_df = proposal_as_text.get_dataframe()
proposal_as_text_df = pd.DataFrame(proposal_as_text_df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Change proposal text (american spelling, phrases, acronyms)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
proposal_as_text_df['words_with_acronyms']=proposal_as_text_df.text.parallel_apply(
    lambda x: change_text(x)
)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
proposal_as_text_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Further process text to produce bag-of-words

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def processDoc(doc):
    norm_words=[]
    for token in doc:
        c=0
        if not token.is_punct and not token.is_stop and not token.is_space and not token.is_digit:
            word = token.lemma_.lower()
            if word in acronyms_plural:
                word = acronyms_plural[word]
            if word not in nlp.Defaults.stop_words:
                for i in numbers_and_symbols:
                    if i in word:
                        c=1
                if c==0 and len(word)>1:
                    norm_words.append(word)
    return norm_words

docs = []
c=0
s = dt.datetime.now()
for doc in nlp.pipe(proposal_as_text_df['words_with_acronyms'], n_process=6, batch_size=10):
    if c % 100 == 0:
        print(c, dt.datetime.now()-s)
    c+=1
    docs.append(processDoc(doc))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Saves the new bag-of-words into the dataframe

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
proposal_as_text_df['final_words'] = docs

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Write the bag-of-words to a folder

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Gets folder path
bow_results = header.FOLDER_BOW_FILES
bow_results_path = bow_results.get_path()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Saves df as parquet file. Function works for remote folders
file_bow = header.NAME_BOW_PARQUET
with bow_results.get_writer(file_bow) as buff:
    buff.write(proposal_as_text_df.to_parquet())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: RAW
# with bow_results.get_download_stream(file_bow) as buff2:
#     f = io.BytesIO(buff2.read())
#     prop = pd.read_parquet(f)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Option: Write out individual bag-of-word files

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
for index in proposal_as_text_df.index:
    # Set output file name
    code = proposal_as_text_df['proposal_code'][index]
    output = f'{bow_results_path}/{code}.bow'
    
    # Write to file
    fout = open(output, 'w')
    fout.write(' '.join(proposal_as_text_df['final_words'][index])+'\n')
    fout.close()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Option: Remove individual bag-of-word files

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import os
os.system(f'rm {bow_results_path}/*.bow')