import numpy as np
import csv
import math


'''def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix'''


def normalize(input_matrix, axis=1):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    sums = input_matrix.sum(axis=axis)
    try:
        assert (np.count_nonzero(sums) == np.shape(sums)[0])  # no row/col should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    if axis == 1:
        new_matrix = input_matrix / sums[:, np.newaxis]
    elif axis == 0:
        new_matrix = input_matrix / sums[np.newaxis, :]
    return new_matrix

       
class Corpus(object):

    """
    A collection of documents.
    """

    def __init__(self, documents_path):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.term_doc_matrix = None 
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)

        self.number_of_documents = 0
        self.vocabulary_size = 0

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]
        
        Update self.number_of_documents
        """
        # #############################
        # your code here
        # #############################
        with open(self.documents_path, 'r') as dest_f:
            data_iter = csv.reader(dest_f, delimiter=' ')
            #corpus = ([data for data in data_iter if data is not ''])
            corpus = []
            i = 1
            for data in data_iter:
                print("line={}".format(data))
                if data[len(data)-1] == '':
                    data = data[:len(data)-1]
                data[0] = data[0].strip()
                corpus.append(data)
                i = i + 1

        '''if 'test.txt' in self.documents_path:
            for i in range(100):
                corpus[i][0] = corpus[i][0][2:]'''

        self.documents = corpus
        self.number_of_documents = len(corpus)

    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """
        # #############################
        # your code here
        # #############################
        vocabulary = set()
        for d in self.documents:
            vocabulary.update(d)

        # print(vocabulary)
        
        self.vocabulary = list(vocabulary)
        self.vocabulary_size = len(vocabulary)

    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        # ############################
        # your code here
        # ############################
        term_doc_counts = np.zeros((len(self.documents), self.vocabulary_size))
        for i in range(len(self.documents)):
            doc_vocabulary = set(self.documents[i])
            for j in range(self.vocabulary_size):
                term_doc_counts[i][j] += self.documents[i].count(self.vocabulary[j])

        self.term_doc_matrix = term_doc_counts


    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize! 
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        # ############################
        # your code here
        # ############################
        self.document_topic_prob = np.random.random_sample((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.random.random_sample((number_of_topics, self.vocabulary_size))
        self.topic_word_prob = normalize(self.topic_word_prob)

    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform 
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob)

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self):
        """ The E-step updates P(z | w, d)
        """
        # print("E step:")
        
        # ############################
        # your code here
        # ############################
        dtp = self.document_topic_prob
        twp = self.topic_word_prob
        tp = self.topic_prob

        for i in range(dtp.shape[0]):
            tp_word = dtp[i][:, np.newaxis]*twp
            norm_tp_word = normalize(tp_word, axis=0)
            tp[i] = norm_tp_word


    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """
        # print("M step:")
        
        # update P(w | z)
        
        # ############################
        # your code here
        # ############################
        dtp = self.document_topic_prob
        twp = self.topic_word_prob
        tp = self.topic_prob
        tdm = self.term_doc_matrix

        for i in range(tp.shape[0]):
            #print(tp[i].transpose())
            updated_dtp = np.matmul(tdm[i], tp[i].transpose())
            dtp[i] = updated_dtp
        self.document_topic_prob = normalize(dtp)

        #print(np.sum(self.document_topic_prob, axis=1))
        # update P(z | d)

        # ############################
        # your code here
        # ############################
        #self.topic_word_prob = np.zeros((number_of_topics, len(self.vocabulary)))
        trans_tp = tp.transpose()
        # print(trans_tp.shape)
        for i in range(trans_tp.shape[0]):
            ith_word_topic_doc = trans_tp[i]
            ith_word_counts = tdm[:, i][:, np.newaxis]
            updated_twp = np.matmul(ith_word_topic_doc, ith_word_counts)
            twp[:, i] = updated_twp[:, 0]

        self.topic_word_prob = normalize(twp)
        # print(self.topic_word_prob)

    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        # ############################
        # your code here
        # ############################
        dtp = self.document_topic_prob
        twp = self.topic_word_prob
        tp = self.topic_prob
        tdm = self.term_doc_matrix

        doc_word_topic_prob = np.log(np.matmul(dtp, twp))
        # print(doc_word_topic_prob.shape)
        likelihood_matrix = tdm * doc_word_topic_prob
        # print(likelihood_matrix.shape)
        new_likelihood = likelihood_matrix.sum()
        self.likelihoods.append(new_likelihood)

        # print(self.likelihoods)
        
        return new_likelihood

    def plsa(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
        print("EM iteration begins...")
        
        # build term-doc matrix
        self.build_term_doc_matrix()
        
        # Create the counter arrays.
        
        # P(z | d, w)
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")

            # ############################
            # your code here
            # ############################
            self.expectation_step()
            self.maximization_step(number_of_topics)
            new_likelihood = self.calculate_likelihood(number_of_topics)
            current_epsilon = np.abs(current_likelihood - new_likelihood)
            print("new_likelihood={}, current_epsilon={}, epsilon={}".format(new_likelihood, current_epsilon, epsilon))
            current_likelihood = new_likelihood
            if current_epsilon <= epsilon:
                break



def main():
    documents_path = 'data/test.txt'
    # documents_path = 'data/DBLP.txt'
    corpus = Corpus(documents_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    print(corpus.vocabulary)
    print("Vocabulary size:" + str(len(corpus.vocabulary)))
    print("Number of documents:" + str(len(corpus.documents)))
    number_of_topics = 2
    max_iterations = 150
    epsilon = 0.001
    corpus.plsa(number_of_topics, max_iterations, epsilon)



if __name__ == '__main__':
    main()
