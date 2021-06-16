#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

@dataclass
class ETACorpus():

    corpus:pd.DataFrame = None
    doc_content_col:str = 'doc_content'
    doc_id_col:str = 'doc_id'
    corpus_db:str = ''
    n_words:int = 5000
    max_df:float = .95
    min_df:float = 2
    stop_words:str = 'english'
    ngram_min:int = 1
    ngram_max:int = 3

    def __post_init__(self):
        self.corpus = self.corpus.reset_index().set_index(self.doc_id_col)
        return self

    def create_bow(self):
        
        print("Initializing Count Engine.")
        self.count_engine = CountVectorizer(
            max_df=self.max_df, 
            min_df=self.min_df,
            lowercase=True, 
            max_features=self.n_words,
            stop_words=self.stop_words, 
            ngram_range=(self.ngram_min, self.ngram_max))
        
        print("Generating Count Model.")
        self.count_model = self.count_engine.fit_transform(self.corpus[self.doc_content_col])
        
        print("Initializing TFIDF Engine.")
        self.tfidf_engine = TfidfTransformer()
        
        print("Generating TFIDF Model.")
        self.tfidf_model = self.tfidf_engine.fit_transform(self.count_model)

        print("Extracting VOCABulary.")
        self.VOCAB = pd.DataFrame(self.count_engine.get_feature_names(), 
                                  columns=['term_str'])
        self.VOCAB.index.name = 'term_id'
        self.termlist = self.VOCAB.term_str.to_list()

        print("Creating Bag of Words table.")
        
        A = pd.DataFrame(self.count_model.toarray()).stack()
        A = (A[A > 0]).to_frame('n')
        A.index.names = ['doc_id','term_id']

        B = pd.DataFrame(self.tfidf_model.toarray()).stack()
        B = (B[B > 0]).to_frame('tfidf')
        B.index.names = ['doc_id','term_id']
        
        self.BOW = pd.merge(A, B, on=['doc_id','term_id'])

        print("Applying stats to VOCAB.")
        self.VOCAB['ngram_len'] = self.VOCAB.apply(lambda x: len(x.term_str.split()), 1)
        self.VOCAB['tfidf_sum'] = self.BOW.groupby(['term_id'])['tfidf'].sum()        
        self.VOCAB['tfidf_mean'] = self.BOW.groupby(['term_id'])['tfidf'].mean()        
        self.VOCAB['corpus_freq'] = self.BOW.groupby(['term_id'])['n'].sum()
        self.VOCAB['prior_prob'] = self.VOCAB['corpus_freq'] / self.VOCAB['corpus_freq'].sum()

        return self # For chaining

    def get_doc_term_matrix(self, bow_col='n'):
        dtm = self.BOW[bow_col].to_frame().unstack(fill_value=0)
        dtm.columns = dtm.columns.droplevel(0)
        return dtm

    def get_count_doc_term_matrix(self):
        return self.get_doc_term_matrix('n')

    def get_tfidf_doc_term_matrix(self):
        return self.get_doc_term_matrix('tfidf')


from abc import ABC, abstractmethod
import scipy.cluster.hierarchy as sch
import prince
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

@dataclass
class AbstractTopicModel(ABC):

    corpus:ETACorpus = None
    n_topics:int = 20
    engine:object = None
    model:object = None
    doc_topic_matrix:pd.DataFrame = None
    term_topic_matrix:pd.DataFrame = None
    topic:pd.DataFrame = None
    hca_pdist_metric:str = 'cosine'
    hca_linkage_method:str = 'ward'

    @abstractmethod
    def generate_model(self):
        pass

    def _create_topic_label(self, topic_id, term_topic_matrix, 
                            ascending=False, n_terms=7):
        topic_label = f"{topic_id}: " + ', '.join(term_topic_matrix[topic_id]\
                        .sort_values(ascending=ascending).head(n_terms)\
                        .index.to_list())
        return topic_label

    def get_term_network(self, n_pairs=1000):
        self.term_net = self.term_topic_matrix.T.corr()\
            .stack().sort_values(ascending=False)\
            .to_frame('corr').head(n_pairs)
        self.term_net.index.names = ['term1', 'term2']
        # self.term_net = self.term_net.rename(columns={0:'corr'})
        self.term_net = self.term_net.reset_index()

        self.term_net = self.term_net.loc[self.term_net.term1 != self.term_net.term2]
        self.term_net['key'] = self.term_net.apply(lambda x: '|'.join(sorted([x.term1, x.term2])), 1)
        self.term_net = self.term_net.drop_duplicates(subset='key')
        self.term_net = self.term_net.drop('key', 1)
        
        W = (self.term_topic_matrix[self.term_topic_matrix > 0]).sum(1)
        self.term_net['term_weight'] = self.term_net.apply(lambda x: W.loc[x.term1] * W.loc[x.term2], 1)
        self.term_net['weight'] = self.term_net.apply(lambda x: x['term_weight'] * x['corr'], 1)
        self.term_net = self.term_net.sort_values('weight', ascending=False)

    def collapse_doc_term_matrix(self, group_col, agg_func='mean'):
        df = self.corpus.corpus.merge(self.doc_topic_matrix, 
            on=self.corpus.doc_id_col).groupby(group_col).agg(agg_func)
        return df
        
    def plot_hca(self, group_col='doc_label'):
        df = self.corpus.corpus.merge(self.doc_topic_matrix, on=self.corpus.doc_id_col)\
            .groupby(group_col).mean()
        labels = df.apply(lambda x: x.name, 1).tolist()
        sims = pdist(df, metric=self.hca_pdist_metric)
        tree = sch.linkage(sims, method=self.hca_linkage_method)
        fig, axes = plt.subplots(figsize=(7, len(labels) / 2))
        dendrogram = sch.dendrogram(tree, labels=labels, orientation="left")
        axes.tick_params(axis='both', which='major', labelsize=15) 

    # Consider binding these features to the topic table
    def get_topic_corr_matrix(self):
        dtmc = self.doc_topic_matrix.cov()
        dtmc = dtmc[dtmc > 0].fillna(0)
        dtmc = np.square(dtmc)
        return dtmc


from sklearn.decomposition import NMF

@dataclass 
class NMFTopicModel(AbstractTopicModel):

    nmf_max_iter:int = 1000
    nmf_alpha:float = .1
    nmf_l1_ratio:float = .5
    nmf_beta_loss:str = 'frobenius' #'kullback-leibler'
    nmf_solver:str = 'cd'
    nmf_random_state:int = 1
    nmf_init:str = 'nndsvd'

    def generate_model(self):

        print("Initializing NMF Engine.")
        self.engine = NMF(
            n_components = self.n_topics, 
            init = self.nmf_init,
            solver = self.nmf_solver, 
            beta_loss = self.nmf_beta_loss, 
            max_iter = self.nmf_max_iter, 
            random_state = self.nmf_random_state, 
            alpha = self.nmf_alpha,
            l1_ratio = self.nmf_l1_ratio)

        print("Generating NMF Model.")
        self.model = self.engine.fit_transform(self.corpus.tfidf_model)

        print("Extracting NMF Doc-Topic Matrix.")
        self.doc_topic_matrix = pd.DataFrame(self.model)
        self.doc_topic_matrix.index.name = 'doc_id'
        self.doc_topic_matrix.columns.name = 'topic_id'

        print("Extracting NMF Term-Topic Matrix.")
        self.term_topic_matrix = pd.DataFrame(self.engine.components_, 
                                                columns=self.corpus.termlist).T
        self.term_topic_matrix.index.name = 'term_str'                                                    
        self.term_topic_matrix.columns.name = 'topic_id'                                                    

        print("Extracting NMF Topics.")
        self.topic = self.doc_topic_matrix.sum().to_frame('preponderance')
        self.topic.index.name = 'topic_id'
        self.topic['label'] = self.topic.apply(lambda x: \
            self._create_topic_label(x.name, self.term_topic_matrix), 1)
        
        return self


from sklearn.decomposition import LatentDirichletAllocation

@dataclass 
class LDATopicModel(AbstractTopicModel):

    lda_max_iter:int = 10
    lda_learning_method:str = 'online'
    lda_learning_offset:float = 50.
    lda_random_state:int = 0

    def generate_model(self):

        print("Initializing LDA Engine.")
        self.engine = LatentDirichletAllocation(
            n_components = self.n_topics, 
            max_iter = self.lda_max_iter,
            learning_method = self.lda_learning_method,
            learning_offset = self.lda_learning_offset, 
            random_state = self.lda_random_state)
        
        print("Generating LDA Model.")
        self.model = self.engine.fit_transform(self.corpus.count_model)

        print("Extracting LDA Doc-Topic Matrix.")
        self.doc_topic_matrix = pd.DataFrame(self.model)
        self.doc_topic_matrix.index.name = 'doc_id'
        self.doc_topic_matrix.columns.name = 'topic_id'

        print("Extracting LDA Term-Topic Matrix.")
        self.term_topic_matrix = pd.DataFrame(self.engine.components_, 
                                                columns=self.corpus.termlist).T
        self.term_topic_matrix.index.name = 'term_str'
        self.term_topic_matrix.columns.name = 'topic_id'

        print("Extracting LDA Topics.")
        self.topic = self.doc_topic_matrix.sum().to_frame('preponderance')
        self.topic.index.name = 'topic_id'
        self.topic['label'] = self.topic.apply(lambda x: \
            self._create_topic_label(x.name, self.term_topic_matrix), 1)

        return self

from sklearn.decomposition import PCA

@dataclass
class PCATopicModel(AbstractTopicModel):
    
    pca_n_components:int = 10
        
    def generate_model(self):

        print("Initializing PCA Engine.")
        self.engine = PCA(n_components=self.pca_n_components)
        
        print("Generating PCA Model.")
        self.model = self.engine.fit_transform(self.corpus.tfidf_model.toarray())

        print("Extracting PCA Doc-Topic Matrix.")
        self.doc_topic_matrix = pd.DataFrame(self.model)
        self.doc_topic_matrix.index.name = 'doc_id'

        print("Extracting PCA Term-Topic Matrix (i.e. Loadings).")
        self.term_topic_matrix = pd.DataFrame(self.engine.components_.T * \
                                        np.sqrt(self.engine.explained_variance_))
        self.term_topic_matrix.index = self.corpus.termlist
        self.term_topic_matrix.index.name = 'term_id'  
        
        print("Extract PCA Topics (i.e. Components).")
        self.topic = pd.DataFrame(self.engine.explained_variance_, columns="preponderance")
        self.topic.index.name = 'topic_id'
        
        self.topic['label'] = self.topic\
            .apply(lambda x: self._create_topic_label(x.name, 
                    self.term_topic_matrix), 1)

        self.topic['label_neg'] = self.topic\
            .apply(lambda x: self._create_topic_label(x.name, 
                    self.term_topic_matrix, ascending=True), 1)

        return self

from sklearn.decomposition import TruncatedSVD as SVD

@dataclass
class SVDTopicModel(AbstractTopicModel):
    
    n_components:int = 10
    n_iter:int = 7
    random_state:int = 42
        
    def generate_model(self):

        print("Initializing SVD Engine.")
        self.engine = SVD(n_components=self.n_components, 
            n_iter=self.n_iter, 
            random_state=self.random_state)
        
        print("Generating SVD Model.")
        self.model = self.engine.fit_transform(self.corpus.tfidf_model.toarray())

        print("Extracting SVD Doc-Topic Matrix.")
        self.doc_topic_matrix = pd.DataFrame(self.model)
        self.doc_topic_matrix.index.name = 'doc_id'

        print("Extracting SVD Term-Topic Matrix (i.e. Loadings).")
        self.term_topic_matrix = pd.DataFrame(self.engine.components_.T * \
                                        np.sqrt(self.engine.explained_variance_))
        self.term_topic_matrix.index = self.corpus.termlist
        self.term_topic_matrix.index.name = 'term_id'  
        
        print("Extract SVD Topics (i.e. Components).")
        self.topic = pd.DataFrame(self.engine.explained_variance_, columns='preponderance')
        self.topic.index.name = 'topic_id'
        
        self.topic['label'] = self.topic\
            .apply(lambda x: self._create_topic_label(x.name, 
                    self.term_topic_matrix), 1)

        self.topic['label_neg'] = self.topic\
            .apply(lambda x: self._create_topic_label(x.name, 
                    self.term_topic_matrix, ascending=True), 1)

        return self
