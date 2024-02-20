import pandas as pd
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer

class TextParser():
    """
    A class to parse a dataframe of project title/ abstracts into a TOKENS dataframe with
    an OHCO index. Also has methods to extract a VOCAB table, although vocabulary
    tables out to be generated at the corpus level.
    
    Sample parameter values:
    
    df- a pandas df with project_id, project_code, and project_title
    """

    # TODO: Make these private
    src_imported:bool = False       
    src_clipped:bool = False
    src_col_suffix:str ='_str'

    join_pat:str = r'\n'
    strip_hyphens:bool = False
    strip_whitespace:bool = False
    verbose:bool = False

    stanford_pos_model:str = "english-bidirectional-distsim.tagger"
    stanford_pos_model_path = None
    ohco_pats = [('sent', r"[.?!;:]+", 'd'),('token', r"[\s',-]+", 'd')]
    _ohco_type:{} = {
        'd': '_num',
        'm': '_id'
    }
        
    def __init__(self, text_df, use_nltk=True):
        """Initialize the object and extract config info. If using NLTK, download resources."""
        self.text_df = text_df
        self.OHCO = [item[0]+self._ohco_type[item[2]] for item in self.ohco_pats]
        self.ohco_names = [item[0] for item in self.ohco_pats]
        self.use_nltk = use_nltk

        if self.use_nltk:
            # Override the last two OHCO items
            self.ohco_pats = [('sent', None, 'nltk'), ('token', None, 'nltk')]
            
            
            
            # Make sure you have the NLTK stuff
            for package in [
                'tokenizers/punkt', 
                'taggers/averaged_perceptron_tagger', 
                'corpora/stopwords', 
                'help/tagsets'
            ]:
                if self.verbose: print("Checking", package)
                try:
                    nltk.data.find(package)
                except IndexError:
                    nltk.download(package)    
    
    def parse_tokens(self):
        books = []
        # Define Parameters
        self.verbose = True
        self.strip_hyphens = True
        self.strip_whitespace = True
        for x in range(len(self.text_df.index)):
        # parse_tokens
            self._parse_text(self.text_df.iloc[x]['project_title_abstract'])

        # Name things
            self.TOKENS['project_code'] = self.text_df.iloc[x]['project_code']
            self.TOKENS = self.TOKENS.reset_index().set_index(['project_code'] + self.OHCO)

        # Add to list
            books.append(self.TOKENS)
        
    # Combine into a single dataframe
        self.CORPUS = pd.concat(books).sort_index()
        self.CORPUS['pos_group'] = self.CORPUS.pos.str[:2]

    # Clean up
        del(books)
        
        print("Done")
        
        return self.CORPUS

        
    def _parse_text(self, text):
        """Convert lines to tokens based on OHCO. Should only be accessed by parse_tokens."""
            # Start with the LINES df
        self.TOKENS = text

            # Walk through each level of the OHCO to build out TOKENS
        for i, level in enumerate(self.OHCO):

            # Define level-specific variables
            parse_type = self.ohco_pats[i][2]
            div_name = self.ohco_pats[i][0]
            div_pat = self.ohco_pats[i][1]
            if i == 0:
                src_div_name = 'line'
            else:
                src_div_name = self.ohco_names[i - 1] 
            src_col = f"{src_div_name}{self.src_col_suffix}"
            dst_col = f"{div_name}{self.src_col_suffix}"

            # By Milestone
            # NOT DONE
            if parse_type == 'm':
                div_lines = self.TOKENS[src_col].str.contains(div_pat, regex=True, case=True) # TODO: Parametize case
                self.TOKENS.loc[div_lines, div_name] = [i+1 for i in range(self.TOKENS.loc[div_lines].shape[0])]
                self.TOKENS[div_name] = self.TOKENS[div_name].ffill()
                self.TOKENS = self.TOKENS.loc[~self.TOKENS[div_name].isna()] 
                self.TOKENS = self.TOKENS.loc[~div_lines] 
                self.TOKENS[div_name] = self.TOKENS[div_name].astype('int')
                self.TOKENS = self.TOKENS.groupby(self.ohco_names[:i+1], group_keys=True)[src_col]\
                    .apply(lambda x: '\n'.join(x)).to_frame(dst_col)

                # print(self.TOKENS[dst_col].str.count(r'\n\n'))
                print(src_col, dst_col)
                print(self.TOKENS.columns)


            # By Delimitter
            # NOT DONE
            elif parse_type == 'd':
                self.TOKENS = self.TOKENS[src_col].str.split(div_pat, expand=True).stack().to_frame(dst_col)
                
            # By NLTK 
            elif parse_type == 'nltk':

                if level == 'sent_num':
                    self.TOKENS = pd.DataFrame(nltk.sent_tokenize(self.TOKENS), columns = ['sent_str'])
                    self.TOKENS
                if level == 'token_num':
                    if self.strip_hyphens == True:
                        self.TOKENS.sent_str = self.TOKENS.sent_str.str.replace(r"-", ' ')
                    if self.strip_whitespace == True:
                        self.TOKENS = self.TOKENS.sent_str\
                                .apply(lambda x: pd.Series(
                                        nltk.pos_tag(nltk.WhitespaceTokenizer().tokenize(x)),
                                        dtype='object'
                                       )
                                    )
                    else:
                        self.TOKENS = self.TOKENS.sent_str\
                                .apply(lambda x: pd.Series(nltk.pos_tag(nltk.word_tokenize(x))))
                    self.TOKENS = self.TOKENS.stack().to_frame('pos_tuple')
                    self.TOKENS['pos'] = self.TOKENS.pos_tuple.apply(lambda x: x[1])
                    self.TOKENS['token_str'] = self.TOKENS.pos_tuple.apply(lambda x: x[0])
                    self.TOKENS['term_str'] = self.TOKENS.token_str.str.lower()   
        
            else:
                raise ValueError(f"Invalid parse option: {parse_type}.")

                # After creating the current OHCO level
            self.TOKENS.index.names = self.OHCO[:i+1]

            # After iterating through the OHCO

            # Not sure if needed anymore
            # self.TOKENS[dst_col] = self.TOKENS[dst_col].str.strip()
            # self.TOKENS[dst_col] = self.TOKENS[dst_col].str.replace(self.join_pat, ' ', regex=True)
            # self.TOKENS = self.TOKENS[~self.TOKENS[dst_col].str.contains(r'^\s*$', regex=True)]

        if not self.use_nltk:
            self.TOKENS['term_str'] = self.TOKENS.token_str.str.replace(r'[\W_]+', '', regex=True).str.lower()  
        else:
            punc_pos = ['$', "''", '(', ')', ',', '--', '.', ':', '``']
            self.TOKENS['term_str'] = self.TOKENS[~self.TOKENS.pos.isin(punc_pos)].token_str\
                .str.replace(r'[\W_]+', '', regex=True).str.lower()  

    def extract_vocab(self, pos_ambig=False, stems='none'):
        """
        Extracts VOCAB, and saves/returns a vocab table. Includes parameters for customization
        that include:
        
        :param bool pos_ambig: Option to include part of speech ambiquity columns. The default is False.
        :param str stems: Option to include columns of various stemmed versions each token. Options include:
        'none' - No stemmed columns added
        'all' - All the following stemmed values are included in the vocab table.
        'porter' - include a column that performs the porter stemming algorithm on the tokens in the table.
        'snowball' - include a column that performs the snowball stemming algorithm on the tokens in the table.
        'lancaster' - include a column that performs the lancaster stemming algorithm on the tokens in the table.
        """
        self.VOCAB = self.CORPUS.term_str.value_counts().to_frame('n').sort_index()
        self.VOCAB.index.name = 'term_str'
        self.VOCAB['n_chars'] = self.VOCAB.index.str.len()
        self.VOCAB['p'] = self.VOCAB.n / self.VOCAB.n.sum()
        self.VOCAB['i'] = -np.log2(self.VOCAB.p)
        if pos_ambig:
            # Implement pos_ambig for vocab table
            self.VOCAB['n_pos_group'] = self.CORPUS[['term_str','pos_group']].value_counts().unstack().count(1)
            self.VOCAB['cat_pos_group'] = self.CORPUS[['term_str','pos_group']].value_counts().to_frame('n').reset_index()\
    .groupby('term_str').pos_group.apply(lambda x: set(x))
            self.VOCAB['n_pos'] = self.CORPUS[['term_str','pos']].value_counts().unstack().count(1)
            self.VOCAB['cat_pos'] = self.CORPUS[['term_str','pos']].value_counts().to_frame('n').reset_index()\
    .groupby('term_str').pos.apply(lambda x: set(x))
        if stems == 'all':
            stemmer1 = PorterStemmer()
            self.VOCAB['stem_porter'] =   self.VOCAB.apply(lambda x: stemmer1.stem(x.name), 1)
            stemmer2 = SnowballStemmer("english")
            self.VOCAB['stem_snowball'] = self.VOCAB.apply(lambda x: stemmer2.stem(x.name), 1)
            stemmer3 = LancasterStemmer()
            self.VOCAB['stem_lancaster'] = self.VOCAB.apply(lambda x: stemmer3.stem(x.name), 1)
            return self.VOCAB
        elif stems == 'porter':
            stemmer1 = PorterStemmer()
            self.VOCAB['stem_porter'] =   self.VOCAB.apply(lambda x: stemmer1.stem(x.name), 1)
            return self.VOCAB
        elif stems == 'snowball':
            stemmer2 = SnowballStemmer("english")
            self.VOCAB['stem_snowball'] = self.VOCAB.apply(lambda x: stemmer2.stem(x.name), 1)
            return self.VOCAB
        elif stems == 'lancaster':
            stemmer3 = LancasterStemmer()
            self.VOCAB['stem_lancaster'] = self.VOCAB.apply(lambda x: stemmer3.stem(x.name), 1)
            return self.VOCAB
        else:
            return self.VOCAB
        
    def corp_to_BOW(self, bag_OHCO=1):
        '''Takes a tokenized data frame and outputs a bag of word representation
        based off a desired OHCO.
    
        :param pd.df token_df: A tokenized df to be transformed into a BOW representation.
        :param int bag_OHCO: An interger representing the level at which the bagging should 
        occur. Options include:
        1 = project_code
        2 = sent_num
        '''
        OHCO = ['project_code', 'sent_num', '']
        bags = OHCO[:bag_OHCO]
        self.Bag = self.CORPUS.groupby(bags +['term_str']).term_str.count().to_frame('n') 
        return self.Bag
    
    def BOW_TFIDF(self, tf_method):
        '''Takes a BOW data frame and outputs df with desired TFIDF values.
    
        :param pd.df bow_df: A BOW df to add desired TFIDF values.
        :param str tf_method: Desired tf_type calculation. Options include:
        sum, max, log, double_norm, raw, binary
        '''
        self.BOW = self.Bag.copy(deep=True)
        DTCM = self.BOW.n.unstack(fill_value=0)
        # Select tf_method
        if tf_method == 'sum':
            TF = DTCM.T / DTCM.T.sum()

        elif tf_method == 'max':
            TF = DTCM.T / DTCM.T.max()
    
        elif tf_method == 'log':
            TF = np.log2(1 + DTCM.T)
    
        elif tf_method == 'raw':
            TF = DTCM.T
    
        elif tf_method == 'double_norm':
            TF = DTCM.T / DTCM.T.max()
    
        elif tf_method == 'binary':
            TF = DTCM.T.astype('bool').astype('int')
   
        TF = TF.T 
        N = DTCM.shape[0]
        DF = DTCM.astype('bool').sum() 
        # Compute IDF with standard method
        IDF = np.log2(N / DF)
        # Compute IDF
        TFIDF = TF * IDF
        # Add to return table and return
        self.BOW['tf'] = TF.stack()
        self.BOW['tfidf'] = TFIDF.stack()
        return self.BOW
    
    
    

    def gather_tokens(self, level=0, grouping_col='term_str', cat_sep=' '):
        """Gather tokens into strings for arbitrary OHCO level."""
        max_level = len(self.OHCO) - 2 # Can't gather tokens at the token level :)
        if level > max_level:
            raise ValueError(f"Level {level} too high. Try between 0 and {max_level}")
        else:
            level_name = self.OHCO[level].split('_')[0]
            idx = self.TOKENS.index.names[:level+1]
            return self.TOKENS.groupby(idx)[grouping_col].apply(lambda x: x.str.cat(sep=cat_sep))\
                .to_frame(f'{level_name}_str')


if __name__ == '__main__':
    pass