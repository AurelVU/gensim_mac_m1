#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Corpus in the `Matrix Market format <https://math.nist.gov/MatrixMarket/formats.html>`_."""

import logging

from gensim import matutils
from gensim.corpora import IndexedCorpus


logger = logging.getLogger(__name__)


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


class MmCorpus(MmReader, IndexedCorpus):
    """Corpus serialized using the `sparse coordinate Matrix Market format
    <https://math.nist.gov/MatrixMarket/formats.html>`_.

    Wrap a term-document matrix on disk (in matrix-market format), and present it
    as an object which supports iteration over the matrix rows (~documents).

    Notes
    -----
    The file is read into memory one document at a time, not the whole matrix at once,
    unlike e.g. `scipy.io.mmread` and other implementations. This allows you to **process corpora which are larger
    than the available RAM**, in a streamed manner.

    Example
    --------
    .. sourcecode:: pycon

        >>> from gensim.corpora.mmcorpus import MmCorpus
        >>> from gensim.test.utils import datapath
        >>>
        >>> corpus = MmCorpus(datapath('test_mmcorpus_with_index.mm'))
        >>> for document in corpus:
        ...     pass

    """
    def __init__(self, fname):
        """

        Parameters
        ----------
        fname : {str, file-like object}
            Path to file in MM format or a file-like object that supports `seek()`
            (e.g. a compressed file opened by `smart_open <https://github.com/RaRe-Technologies/smart_open>`_).

        """
        # avoid calling super(), too confusing
        IndexedCorpus.__init__(self, fname)
        matutils.MmReader.__init__(self, fname)

    def __iter__(self):
        """Iterate through all documents.

        Yields
        ------
        list of (int, numeric)
            Document in the `sparse Gensim bag-of-words format <intro.rst#core-concepts>`__.

        Notes
        ------
        The total number of vectors returned is always equal to the number of rows specified in the header.
        Empty documents are inserted and yielded where appropriate, even if they are not explicitly stored in the
        (sparse) Matrix Market file.

        """
        for doc_id, doc in super(MmCorpus, self).__iter__():
            yield doc  # get rid of doc id, return the sparse vector only

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, progress_cnt=1000, metadata=False):
        """Save a corpus to disk in the sparse coordinate Matrix Market format.

        Parameters
        ----------
        fname : str
            Path to file.
        corpus : iterable of list of (int, number)
            Corpus in Bow format.
        id2word : dict of (int, str), optional
            Mapping between word_id -> word. Used to retrieve the total vocabulary size if provided.
            Otherwise, the total vocabulary size is estimated based on the highest feature id encountered in `corpus`.
        progress_cnt : int, optional
            How often to report (log) progress.
        metadata : bool, optional
            Writes out additional metadata?

        Warnings
        --------
        This function is automatically called by :class:`~gensim.corpora.mmcorpus.MmCorpus.serialize`, don't
        call it directly, call :class:`~gensim.corpora.mmcorpus.MmCorpus.serialize` instead.

        Example
        -------
        .. sourcecode:: pycon

            >>> from gensim.corpora.mmcorpus import MmCorpus
            >>> from gensim.test.utils import datapath
            >>>
            >>> corpus = MmCorpus(datapath('test_mmcorpus_with_index.mm'))
            >>>
            >>> MmCorpus.save_corpus("random", corpus)  # Do not do it, use `serialize` instead.
            [97, 121, 169, 201, 225, 249, 258, 276, 303]

        """
        logger.info("storing corpus in Matrix Market format to %s", fname)
        num_terms = len(id2word) if id2word is not None else None
        return matutils.MmWriter.write_corpus(
            fname, corpus, num_terms=num_terms, index=True, progress_cnt=progress_cnt, metadata=metadata
        )
