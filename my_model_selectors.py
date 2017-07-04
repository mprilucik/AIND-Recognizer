import math
import statistics
import warnings
import sys

import numpy as np
from hmmlearn.hmm import GaussianHMM

from sklearn.model_selection import KFold
from asl_utils import combine_sequences




class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    
    where L is the likelihood of the 
fitted model, p is the number of parameters,
    and N is the number of data points. The term -2 log L decreases with
    increasing model complexity (more parameters), whereas the penalties 2p or
    p log N increase with increasing complexity. The BIC applies a larger penalty
    when N > e2 = 7:4.    
    
    Model selection: The lower the BIC value the better the model    
    
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        min_bic = float("inf")
        min_Model = None

        N = len(self.X)
        logN = np.log(N)
        
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = GaussianHMM(n_components=n_components, n_iter=1000).fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)
                bic = -2*logL + n_components * logN
                if (bic < min_bic):
                    min_bic = bic
                    min_Model = model
            except:
                pass # passing models that cannot be trained/scored
#                e = sys.exc_info()[0]
#                print ('error', e, str(e))
                
#        print ('model', min_Model)
#        print ('bic', min_bic)
        return min_Model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def select(self):
        """
        Tip:
        In order to run hmmlearn training using the X,lengths tuples on the new folds, 
        subsets must be combined based on the indices given for the folds. 
        A helper utility has been provided in the asl_utils module named combine_sequences for this purpose.
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        word_sequences = self.words[self.this_word]
        
        max_LogL = float("-inf")
        max_Model = None
        
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            split_method = KFold(n_splits=n_components)
            try:
                for cv_train_idx, cv_test_idx in split_method.split(word_sequences):
#                    print("n_components: {} Train fold indices:{} Test fold indices:{}".format(n_components, len(cv_train_idx), len(cv_test_idx)))  # view indices of the folds
                    train_X, train_lengths = combine_sequences(cv_train_idx, word_sequences)
                    model = GaussianHMM(n_components=n_components, n_iter=1000).fit(train_X, train_lengths)
                    test_X, test_lengths = combine_sequences(cv_test_idx, word_sequences)
                    logL = model.score(test_X, test_lengths)
                    if (logL > max_LogL):
                        max_LogL = logL
                        max_Model = model
            except:
                pass # passing models that cannot be trained/scored
#                e = sys.exc_info()[0]
#                print ('error', e, str(e))
                
#        print ('model', max_Model)
#        print ('logL', max_LogL)
        return max_Model