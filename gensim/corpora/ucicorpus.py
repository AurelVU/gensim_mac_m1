#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2012 Jonathan Esterhazy <jonathan.esterhazy at gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""Corpus in `UCI format <http://archive.ics.uci.edu/ml/datasets/Bag+of+Words>`_."""


import logging
from collections import defaultdict

from gensim import utils
from gensim.corpora import Dictionary
from gensim.corpora import IndexedCorpus


class MmReader(object):
    """
    MmReader(input, transposed=True)
    Matrix market file reader (fast Cython version), used internally in :class:`~gensim.corpora.mmcorpus.MmCorpus`.

        Wrap a term-document matrix on disk (in matrix-market format), and present it
        as an object which supports iteration over the rows (~documents).

        Attributes
        ----------
        num_docs : int
            Number of documents in the market matrix file.
        num_terms : int
            Number of terms.
        num_nnz : int
            Number of non-zero terms.

        Notes
        -----
        Note that the file is read into memory one document at a time, not the whole matrix at once
        (unlike e.g. `scipy.io.mmread` and other implementations).
        This allows us to process corpora which are larger than the available RAM.
    """

    def docbyoffset(self, offset):  # real signature unknown; restored from __doc__
        """
        MmReader.docbyoffset(self, offset)
        Get the document at file offset `offset` (in bytes).

                Parameters
                ----------
                offset : int
                    File offset, in bytes, of the desired document.

                Returns
                ------
                list of (int, str)
                    Document in sparse bag-of-words format.
        """
        pass

    def skip_headers(self, input_file):  # real signature unknown; restored from __doc__
        """
        MmReader.skip_headers(self, input_file)
        Skip file headers that appear before the first document.

                Parameters
                ----------
                input_file : iterable of str
                    Iterable taken from file in MM format.
        """
        pass

    def __init__(self, *args, **kwargs):  # real signature unknown
        """
        Parameters
                ----------
                input : {str, file-like object}
                    Path to the input file in MM format or a file-like object that supports `seek()`
                    (e.g. smart_open objects).

                transposed : bool, optional
                    Do lines represent `doc_id, term_id, value`, instead of `term_id, doc_id, value`?
        """
        pass

    def __iter__(self, *args, **kwargs):  # real signature unknown
        """
        Iterate through all documents in the corpus.

                Notes
                ------
                Note that the total number of vectors returned is always equal to the number of rows specified
                in the header: empty documents are inserted and yielded where appropriate, even if they are not explicitly
                stored in the Matrix Market file.

                Yields
                ------
                (int, list of (int, number))
                    Document id and document in sparse bag-of-words format.
        """
        pass

    def __len__(self, *args, **kwargs):  # real signature unknown
        """ Get the corpus size: total number of documents. """
        pass

    @staticmethod  # known case of __new__
    def __new__(*args, **kwargs):  # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __reduce__(self, *args, **kwargs):  # real signature unknown
        """ MmReader.__reduce_cython__(self) """
        pass

    def __setstate__(self, *args, **kwargs):  # real signature unknown
        """ MmReader.__setstate_cython__(self, __pyx_state) """
        pass

    def __str__(self, *args, **kwargs):  # real signature unknown
        """ Return str(self). """
        pass

    input = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """input: object"""

    num_docs = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """num_docs: 'long long'"""

    num_nnz = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """num_nnz: 'long long'"""

    num_terms = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """num_terms: 'long long'"""

    transposed = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """transposed: 'bool'"""


from gensim.matutils import MmWriter


logger = logging.getLogger(__name__)


class UciReader(MmReader):
    """Reader of UCI format for :class:`gensim.corpora.ucicorpus.UciCorpus`."""
    def __init__(self, input):
        """

        Parameters
        ----------
        input : str
            Path to file in UCI format.

        """

        logger.info('Initializing corpus reader from %s', input)

        self.input = input

        with utils.open(self.input, 'rb') as fin:
            self.num_docs = self.num_terms = self.num_nnz = 0
            try:
                self.num_docs = int(next(fin).strip())
                self.num_terms = int(next(fin).strip())
                self.num_nnz = int(next(fin).strip())
            except StopIteration:
                pass

        logger.info(
            "accepted corpus with %i documents, %i features, %i non-zero entries",
            self.num_docs, self.num_terms, self.num_nnz
        )

    def skip_headers(self, input_file):
        """Skip headers in `input_file`.

        Parameters
        ----------
        input_file : file
            File object.

        """
        for lineno, _ in enumerate(input_file):
            if lineno == 2:
                break


class UciWriter(MmWriter):
    """Writer of UCI format for :class:`gensim.corpora.ucicorpus.UciCorpus`.

    Notes
    ---------
    This corpus format is identical to `Matrix Market format<http://math.nist.gov/MatrixMarket/formats.html>,
    except for different file headers. There is no format line, and the first three lines of the file
    contain `number_docs`, `num_terms`, and `num_nnz`, one value per line.

    """
    MAX_HEADER_LENGTH = 20  # reserve 20 bytes per header value
    FAKE_HEADER = utils.to_utf8(' ' * MAX_HEADER_LENGTH + '\n')

    def write_headers(self):
        """Write blank header lines. Will be updated later, once corpus stats are known."""
        for _ in range(3):
            self.fout.write(self.FAKE_HEADER)

        self.last_docno = -1
        self.headers_written = True

    def update_headers(self, num_docs, num_terms, num_nnz):
        """Update headers with actual values."""
        offset = 0
        values = [utils.to_utf8(str(n)) for n in [num_docs, num_terms, num_nnz]]

        for value in values:
            if len(value) > len(self.FAKE_HEADER):
                raise ValueError('Invalid header: value too large!')
            self.fout.seek(offset)
            self.fout.write(value)
            offset += len(self.FAKE_HEADER)

    @staticmethod
    def write_corpus(fname, corpus, progress_cnt=1000, index=False):
        """Write corpus in file.

        Parameters
        ----------
        fname : str
            Path to output file.
        corpus: iterable of list of (int, int)
            Corpus in BoW format.
        progress_cnt : int, optional
            Progress counter, write log message each `progress_cnt` documents.
        index : bool, optional
            If True - return offsets, otherwise - nothing.

        Return
        ------
        list of int
            Sequence of offsets to documents (in bytes), only if index=True.

        """
        writer = UciWriter(fname)
        writer.write_headers()

        num_terms, num_nnz = 0, 0
        docno, poslast = -1, -1
        offsets = []
        for docno, bow in enumerate(corpus):
            if docno % progress_cnt == 0:
                logger.info("PROGRESS: saving document #%i", docno)
            if index:
                posnow = writer.fout.tell()
                if posnow == poslast:
                    offsets[-1] = -1
                offsets.append(posnow)
                poslast = posnow

            vector = [(x, int(y)) for (x, y) in bow if int(y) != 0]  # integer count, not floating weights
            max_id, veclen = writer.write_vector(docno, vector)
            num_terms = max(num_terms, 1 + max_id)
            num_nnz += veclen
        num_docs = docno + 1

        if num_docs * num_terms != 0:
            logger.info(
                "saved %ix%i matrix, density=%.3f%% (%i/%i)",
                num_docs, num_terms, 100.0 * num_nnz / (num_docs * num_terms),
                num_nnz, num_docs * num_terms
            )

        # now write proper headers, by seeking and overwriting the spaces written earlier
        writer.update_headers(num_docs, num_terms, num_nnz)

        writer.close()
        if index:
            return offsets


class UciCorpus(UciReader, IndexedCorpus):
    """Corpus in the UCI bag-of-words format."""
    def __init__(self, fname, fname_vocab=None):
        """
        Parameters
        ----------
        fname : str
            Path to corpus in UCI format.
        fname_vocab : bool, optional
            Path to vocab.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.corpora import UciCorpus
            >>> from gensim.test.utils import datapath
            >>>
            >>> corpus = UciCorpus(datapath('testcorpus.uci'))
            >>> for document in corpus:
            ...     pass

        """
        IndexedCorpus.__init__(self, fname)
        UciReader.__init__(self, fname)

        if fname_vocab is None:
            fname_vocab = utils.smart_extension(fname, '.vocab')

        self.fname = fname
        with utils.open(fname_vocab, 'rb') as fin:
            words = [word.strip() for word in fin]
        self.id2word = dict(enumerate(words))

        self.transposed = True

    def __iter__(self):
        """Iterate over the corpus.

        Yields
        ------
        list of (int, int)
            Document in BoW format.

        """
        for docId, doc in super(UciCorpus, self).__iter__():
            yield doc  # get rid of docId, return the sparse vector only

    def create_dictionary(self):
        """Generate :class:`gensim.corpora.dictionary.Dictionary` directly from the corpus and vocabulary data.

        Return
        ------
        :class:`gensim.corpora.dictionary.Dictionary`
            Dictionary, based on corpus.

        Examples
        --------

        .. sourcecode:: pycon

            >>> from gensim.corpora.ucicorpus import UciCorpus
            >>> from gensim.test.utils import datapath
            >>> ucc = UciCorpus(datapath('testcorpus.uci'))
            >>> dictionary = ucc.create_dictionary()

        """
        dictionary = Dictionary()

        # replace dfs with defaultdict to avoid downstream KeyErrors
        # uci vocabularies may contain terms that are not used in the document data
        dictionary.dfs = defaultdict(int)

        dictionary.id2token = self.id2word
        dictionary.token2id = utils.revdict(self.id2word)

        dictionary.num_docs = self.num_docs
        dictionary.num_nnz = self.num_nnz

        for docno, doc in enumerate(self):
            if docno % 10000 == 0:
                logger.info('PROGRESS: processing document %i of %i', docno, self.num_docs)

            for word, count in doc:
                dictionary.dfs[word] += 1
                dictionary.num_pos += count

        return dictionary

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, progress_cnt=10000, metadata=False):
        """Save a corpus in the UCI Bag-of-Words format.

        Warnings
        --------
        This function is automatically called by :meth`gensim.corpora.ucicorpus.UciCorpus.serialize`,
        don't call it directly, call :meth`gensim.corpora.ucicorpus.UciCorpus.serialize` instead.

        Parameters
        ----------
        fname : str
            Path to output file.
        corpus: iterable of iterable of (int, int)
            Corpus in BoW format.
        id2word : {dict of (int, str), :class:`gensim.corpora.dictionary.Dictionary`}, optional
            Mapping between words and their ids. If None - will be inferred from `corpus`.
        progress_cnt : int, optional
            Progress counter, write log message each `progress_cnt` documents.
        metadata : bool, optional
            THIS PARAMETER WILL BE IGNORED.

        Notes
        -----
        There are actually two files saved: `fname` and `fname.vocab`, where `fname.vocab` is the vocabulary file.

        """
        if id2word is None:
            logger.info("no word id mapping provided; initializing from corpus")
            id2word = utils.dict_from_corpus(corpus)
            num_terms = len(id2word)
        elif id2word:
            num_terms = 1 + max(id2word)
        else:
            num_terms = 0

        # write out vocabulary
        fname_vocab = utils.smart_extension(fname, '.vocab')
        logger.info("saving vocabulary of %i words to %s", num_terms, fname_vocab)
        with utils.open(fname_vocab, 'wb') as fout:
            for featureid in range(num_terms):
                fout.write(utils.to_utf8("%s\n" % id2word.get(featureid, '---')))

        logger.info("storing corpus in UCI Bag-of-Words format: %s", fname)

        return UciWriter.write_corpus(fname, corpus, index=True, progress_cnt=progress_cnt)
