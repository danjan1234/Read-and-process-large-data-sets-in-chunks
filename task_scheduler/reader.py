"""
This module is designated to file reading in a variety of different scenarios

Author: Justin Duan
Time: 2018/09/19 03:29PM
"""

import os
import glob
import re
import pandas as pd
import numpy as np


class Reader:
    """
    The following file reading scenarios will be considered:
        1) single file broken into chunks of fixed or variable sizes (chunk size controlled by specific columns)
        2) multiple files. Each file is read into memory as a whole
        3) multiple files. Each file is broken to chunks of fixed or variable sizes
        4) multiple files. Files are grouped to different groups with each group read into memory simultaneously with
           no file breaking into chunks
        5) multiple files. Same as 4) but files within each group are broken to chunks of fixed or variable sizes

    Note: in case that multiple files are requested to be broken to chunks. The program will try to break each file to
    chunks and iterate through the chunks before iterating through different files. In other word, the chunk iteration
    is the inner loop while the file iteration is the outer loop. The looping order idea can be explained by the
    following sketch:
                            |c1 |c2 |c3 |c4 |... (inner loop)
                  ------------------------------------------
                  f1a & f1b |   |   |   |   |
                  f2a & f2b |   |   |   |   |
                      .     |   |   |   |   |
                      .     |   |   |   |   |
                      .     |   |   |   |   |
                 (outer loop)
    """

    def __init__(self, file_path, filegroup_groupby_pattern=None, filegroup_sortby_pattern=None, dtype=None,
                 chunk_size=-1, chunk_by=None, chunking_inmemory=False, kwargs={}, log=None):
        """
        Arguments:
            file_path: str or list of strs
                file_path has the following options:
                    1) A pd.DataFrame object
                    2) A single file (.txt, .h5, .csv, .bin)
                    3) A list of file paths
                    4) Directory path + file name regular expression patten, e.g., [^.]*.h5 matches all h5 files
            filegroup_groupby_pattern: bool, default=None
                Regular expression pattern used to group files into different sub-groups if multiple files are
                requested to be processed
            filegroup_sortby_pattern: bool, default=None
                Used in conjunction with filegroup_groupby_pattern. This parameter specifies the regular expression
                pattern to sort the files within sub-groups
            dtype: dict or numpy.dtype, default None
                If a dictionary is passed, it specifies the column data types of text files (csv, txt). If numpy.dtypes
                is passed, it specifies the column names and data types of binary files
            chunk_size: int, default=-1
                The number of rows per chunk to read
            chunk_by: column name, default=None
                This parameter specifies the column used as the guild line to split the file(s) to chunks. If not None,
                two read passes will be performed. The first pass scans the file and creates a summary based on the
                specified column. The summary specifies the variable chunk sizes while reading in the second pass.
                This ensures rows with same values in the specified columns are read in as a whole without the risk of
                being split across different chunks
                Note: if both chunk_size == -1 and chunk_size_by is None, then file(s) will not be broken to chunks
            chunking_inmemory: bool, default=False
                If true, the file(s) will be read into memory as a whole and split into chunks afterward. This option
                has no effect if a pd.DataFrame object is passed
            kwargs: dict, default={}
                Keyword argument passed to file reading functions (pd.read_csv, pd.read_hdf, np.fromfile, etc.)
            log: an instance to a logging object, default=None
        """
        self._file_path = file_path
        self._dir_path = ""
        self._filegroup_groupby_pattern = filegroup_groupby_pattern
        self._filegroup_sortby_pattern = filegroup_sortby_pattern
        self._dtype = dtype
        self._chunk_size = chunk_size
        self._chunk_by = chunk_by
        self._chunking_inmemory = chunking_inmemory
        self._kwargs = kwargs
        self._log = log

    def _log_message(self, msg):
        """
        Log the message
        """
        try:
            self._log.put(msg)
        except AttributeError:
            pass

    def _filter_files(self, file_paths, file_pattern, directory):
        """
        Filter files based on the given pattern using regular expression
        """
        filtered_paths = []
        for file_path in file_paths:
            if os.path.isdir(file_path):
                continue
            file_name = os.path.relpath(file_path, directory)
            if re.search(file_pattern, file_name) is None:
                continue
            filtered_paths.append(file_path)
        return filtered_paths

    def _get_files(self, file_path):
        """
        Search and return a list of file paths to be processed
        """
        try:
            if '*' in file_path:
                directory, file_pattern = os.path.split(file_path)
                file_paths = glob.glob(os.path.join(directory, "**", "*.*"), recursive=True)
                return self._filter_files(file_paths, file_pattern, directory)
            else:
                return [file_path]
        except AttributeError:
            self._log_message("The file path is not valid: {}".format(file_path))

    def _get_filegroups(self, file_paths, filegroup_groupby_pattern, filegroup_sortby_pattern):
        """
        Group the input file paths to sub-groups based on the group-by and sort-by patterns
        """
        # Corner cases
        if isinstance(self._file_path, pd.DataFrame):
            return [[file_paths]]
        elif len(file_paths) == 1 or filegroup_groupby_pattern is None or filegroup_sortby_pattern is None:
            return [[x] for x in file_paths]

        dic = {}
        for file in file_paths:
            key = re.search(filegroup_groupby_pattern, file)
            identifier = re.search(filegroup_sortby_pattern, file)
            try:
                key, identifier = key.group(), identifier.group()
                if key not in dic:
                    dic[key] = [(identifier, file)]
                else:
                    dic[key].append((identifier, file))
            except AttributeError:
                self._log_message("AttributeError: Key or identifier is not found in file".format(file))

        for key in dic:
            dic[key] = sorted(dic[key], key=lambda x: x[0])
            dic[key] = [x[1] for x in dic[key]]

        return list(dic.values())

    def chunk_iterator(self):
        """
        Create a chunk iterator. Note the yield return is a list of tuples and the elements within each tuple are:
        file_path (relative without the directory prefix), chunk_id, DataFrame
        """
        if isinstance(self._file_path, pd.DataFrame):
            file_paths = self._file_path
            self._dir_path = None
        elif hasattr(self._file_path, '__iter__') and not isinstance(self._file_path, str):
            file_paths = self._file_path
            if len(file_paths) == 0:
                return
            elif len(file_paths) > 1:
                self._dir_path = self._common_prefix_k_strs(file_paths)
            else:
                self._dir_path, _ = os.path.split(self._file_path[0])
        else:
            file_paths = self._get_files(self._file_path)
            self._dir_path, _ = os.path.split(self._file_path)
        file_groups = self._get_filegroups(file_paths, self._filegroup_groupby_pattern, self._filegroup_sortby_pattern)
        chunk_sizes = self._chunk_size

        # File group loop
        for filegroup_id, file_group in enumerate(file_groups):
            # File sub-group loop
            file_iterators = []
            for filepath_id, file_path in enumerate(file_group):
                if self._chunk_by is not None:
                    self._log_message("Build the summary based on the split-by column ...")
                    chunk_sizes = self._get_variable_chunk_sizes(self._single_file_chunk_iterator(file_path,
                                                                                                  self._chunk_size))
                file_iterators.append(self._single_file_chunk_iterator(file_path, chunk_sizes))
            # Chunk iteration
            try:
                chunk_id = 0
                while True:
                    result = []
                    for filepath_id, file_path in enumerate(file_group):
                        df = next(file_iterators[filepath_id])
                        try:
                            rel_filepath = os.path.relpath(file_path, self._dir_path)
                        except TypeError:
                            rel_filepath = None
                        result.append((rel_filepath, chunk_id, df))
                    chunk_id += 1
                    yield result
            except StopIteration:
                pass

    def _single_file_chunk_iterator(self, file_path, chunk_sizes):
        """
        Create a chunk iterator associated with single files, i.e., break the file to chunks
        """
        if isinstance(file_path, pd.DataFrame):
            yield from self._df_iterator(file_path, chunk_sizes)
        elif file_path[-4:] == '.bin':
            yield from self._bin_iterator(file_path, chunk_sizes)
        elif file_path[-4:] in {'.txt', '.csv'}:
            yield from self._txt_iterator(file_path, chunk_sizes)
        elif file_path[-3:] == '.h5':
            yield from self._hdf_iterator(file_path, chunk_sizes)

    def _bin_iterator(self, file_path, chunk_sizes):
        """
        Create a chunk iterator associated with single binary files. Raise StopIteration if the file is exhausted
        """
        with open(file_path, 'rb') as f:
            if not hasattr(chunk_sizes, '__iter__') and chunk_sizes < 0:
                yield pd.DataFrame(np.fromfile(f, dtype=self._dtype, **self._kwargs))
            elif self._chunking_inmemory:
                df = pd.DataFrame(np.fromfile(f, dtype=self._dtype, **self._kwargs))
                yield from self._df_iterator(df, chunk_sizes)
            else:
                chunk_id = 0
                chunk_size = chunk_sizes
                while True:
                    if hasattr(chunk_sizes, '__iter__'):
                        if chunk_id == len(chunk_sizes):
                            raise StopIteration
                        chunk_size = chunk_sizes[chunk_id]
                        chunk_id += 1
                    rtn = pd.DataFrame(np.fromfile(f, dtype=self._dtype, count=chunk_size, **self._kwargs))
                    if len(rtn) == 0:
                        raise StopIteration
                    yield rtn

    def _txt_iterator(self, file_path, chunk_sizes):
        """
        Create a chunk iterator associated with single text files. Raise StopIteration if the file is exhausted
        """
        if not hasattr(chunk_sizes, '__iter__') and chunk_sizes < 0:
            yield pd.read_csv(file_path, dtype=self._dtype, **self._kwargs)
        elif self._chunking_inmemory:
            df = pd.read_csv(file_path, dtype=self._dtype, **self._kwargs)
            yield from self._df_iterator(df, chunk_sizes)
        else:
            chunk_id = 0
            chunk_size = chunk_sizes
            reader = pd.read_csv(file_path, iterator=True, dtype=self._dtype, **self._kwargs)
            while True:
                if hasattr(chunk_sizes, '__iter__'):
                    if chunk_id == len(chunk_sizes):
                        raise StopIteration
                    chunk_size = chunk_sizes[chunk_id]
                    chunk_id += 1
                yield reader.get_chunk(chunk_size)

    def _hdf_iterator(self, file_path, chunk_sizes):
        """
        Create a chunk iterator associated with single hdf files. Raise StopIteration if the file is exhausted
        """
        if not hasattr(chunk_sizes, '__iter__') and chunk_sizes < 0:
            yield pd.read_hdf(file_path, **self._kwargs)
        elif self._chunking_inmemory:
            df = pd.read_hdf(file_path, **self._kwargs)
            yield from self._df_iterator(df, chunk_sizes)
        else:
            start = 0
            chunk_id = 0
            chunk_size = chunk_sizes
            while True:
                if self._chunking_inmemory:
                    pass
                else:
                    if hasattr(chunk_sizes, '__iter__'):
                        if chunk_id == len(chunk_sizes):
                            raise StopIteration
                        chunk_size = chunk_sizes[chunk_id]
                        chunk_id += 1
                    rtn = pd.read_hdf(file_path, start=start, stop=start + chunk_size, **self._kwargs)
                    if len(rtn) == 0:
                        raise StopIteration
                    yield rtn
                    start += chunk_size

    def _df_iterator(self, df, chunk_sizes):
        """
        Create a chunk iterator associated with DataFrame objects. Raise StopIteration if the file is exhausted
        """
        if not hasattr(chunk_sizes, '__iter__') and chunk_sizes < 0:
            yield df
        else:
            start = 0
            chunk_id = 0
            chunk_size = chunk_sizes
            while True:
                if hasattr(chunk_sizes, '__iter__'):
                    if chunk_id == len(chunk_sizes):
                        raise StopIteration
                    chunk_size = chunk_sizes[chunk_id]
                    chunk_id += 1
                rtn = df.loc[start:start + chunk_size - 1, :]
                if len(rtn) == 0:
                    raise StopIteration
                yield rtn
                start += chunk_size

    def _summarize_by_column(self, df, by):
        """
        Create a summary based on specified column
        """
        df.reset_index(inplace=True)
        df = pd.pivot_table(df, index=by, values='index', aggfunc=(len, min))
        return df

    def _merge(self, dfs):
        """
        Merge k DataFrames
        """
        if dfs is None or len(dfs) == 0:
            return

        return self._merge_helper(dfs, 0, len(dfs) - 1)

    def _merge_helper(self, dfs, start, end):
        """
        Merge k DataFrame Helper function
        """
        if start == end:
            return dfs[start]

        mid = start + (end - start) // 2
        left = self._merge_helper(dfs, start, mid)
        right = self._merge_helper(dfs, mid + 1, end)
        return self._merge_two(left, right)

    @staticmethod
    def _merge_two(left, right):
        """
        Merge two DataFrames
        """
        df = pd.merge(left, right, how='outer', left_index=True,
                      right_index=True, suffixes=('_l', '_r'))
        df[['len_l', 'len_r']] = df[['len_l', 'len_r']].fillna(0)
        df[['min_l', 'min_r']] = df[['min_l', 'min_r']].fillna(float('inf'))
        df['len'] = df['len_l'] + df['len_r']
        df['min'] = np.minimum(df['min_l'], df['min_r'])
        return df[['len', 'min']].astype(np.intc)

    def _get_variable_chunk_sizes(self, file_iterator):
        """
        Computer the variable chunk sizes based on specified column. Return a numpy 1-D array
        """
        self._log_message("Build the summary based on the split-by column ...")
        dfs = []
        try:
            while True:
                df = self._summarize_by_column(next(file_iterator), by=self._chunk_by)
                dfs.append(df)
        except StopIteration:
            return self._merge(dfs)['len'].values

    @classmethod
    def _common_prefix(cls, str1, str2):
        str1, str2 = str1.split('\\'), str2.split('\\')
        if str1 is None or str2 is None or len(str1) == 0 or len(str2) == 0:
            return ""
        i = 0
        while i < min(len(str1), len(str2)):
            if str1[i] != str2[i]:
                break
            i += 1
        return "\\".join(str1[:i])

    @classmethod
    def _common_prefix_k_strs(cls, str_list):
        """Compute the common prefix of k strings"""
        if str_list is None or len(str_list) == 0:
            return ""
        return cls._common_prefix_k_strs_helper(str_list, 0, len(str_list) - 1)

    @classmethod
    def _common_prefix_k_strs_helper(cls, str_list, start, end):
        """Helper function for _common_prefix_k_strs"""
        if start == end:
            return str_list[start]

        mid = start + (end - start) // 2
        left = cls._common_prefix_k_strs_helper(str_list, start, mid)
        right = cls._common_prefix_k_strs_helper(str_list, mid + 1, end)
        return cls._common_prefix(left, right)


if __name__ == '__main__':
    # A few test runs
    print("\nTest file listing and grouping:")
    file_path = r'..\test_data\test_data_set\[^.]*.[txth5]+'
    reader = Reader(file_path, chunk_size=3, kwargs={'sep': r'\t'})
    file_paths = reader._get_files(file_path)
    print(file_paths)
    file_groups = reader._get_filegroups(file_paths, reader._filegroup_groupby_pattern,
                                         reader._filegroup_sortby_pattern)
    print("All matching files found: \n{}\n".format(file_paths))
    print("Apply grouping (none): \n{}\n".format(file_groups))
    reader = Reader(file_path, chunk_size=3, kwargs={'sep': r'\t'}, filegroup_groupby_pattern="_set\d+",
                    filegroup_sortby_pattern="\d\w+.txt")
    file_groups = reader._get_filegroups(reader._get_files(reader._file_path), reader._filegroup_groupby_pattern,
                                         reader._filegroup_sortby_pattern)
    print("Apply grouping (group by set, sort by order): \n{}\n".format(file_groups))
    reader = Reader(file_path, chunk_size=3, kwargs={'sep': r'\t'}, filegroup_groupby_pattern="_set\d+",
                    filegroup_sortby_pattern="\D+.txt")
    file_groups = reader._get_filegroups(reader._get_files(reader._file_path), reader._filegroup_groupby_pattern,
                                         reader._filegroup_sortby_pattern)
    print("Apply grouping (group by set, sort by letter): \n{}\n".format(file_groups))
    reader = Reader(file_path, chunk_size=3, kwargs={'sep': r'\t'}, filegroup_groupby_pattern="\d\w+.txt",
                    filegroup_sortby_pattern="_set\d+")
    file_groups = reader._get_filegroups(reader._get_files(reader._file_path), reader._filegroup_groupby_pattern,
                                         reader._filegroup_sortby_pattern)
    print("Apply grouping (group by order, sort by set): \n{}\n".format(file_groups))

    def check_iterator(it):
        try:
            while True:
                tmp = next(it)
                print("The length of the read chunk is : {}".format(len(tmp)))
        except StopIteration:
            print("Reached the end of the file.")

    print("\nTest text file iterator:")
    file_path = r'..\test_data\test_data_set\test_set1 - 1st.txt'
    reader = Reader(file_path, chunk_size=3, kwargs={'sep': "\t"})
    check_iterator(reader._txt_iterator(file_path, chunk_sizes=3))

    print("\nTest hdf file (fixed) iterator:")
    file_path = r'..\test_data\test_data_set\test_set1 - 1st_fixed.h5'
    reader = Reader(file_path, chunk_size=3)
    check_iterator(reader._hdf_iterator(file_path, chunk_sizes=3))

    print("\nTest hdf file (table) iterator:")
    file_path = r'..\test_data\test_data_set\test_set1 - 1st_table.h5'
    reader = Reader(file_path, chunk_size=3, chunking_inmemory=True)
    check_iterator(reader._hdf_iterator(file_path, chunk_sizes=3))

    print("\nTest in-memory iterator:")
    file_path = r'..\test_data\test_data_set\test_set1 - 1st_table.h5'
    reader = Reader(file_path, chunk_size=3, chunking_inmemory=True)
    check_iterator(reader._hdf_iterator(file_path, chunk_sizes=3))

    def check_iterator_multi(it):
        try:
            while True:
                tmp = next(it)
                for x in tmp:
                    print("The length of the read chunk is : {2}. From file '{0}' w/ chunk_id = {1}".format(x[0], x[1],
                                                                                                            len(x[2])))
        except StopIteration:
            print("Reached the end of the file.")

    print("\nTest iteration over multiple files:")
    file_path = r'..\test_data\test_data_set\[^.]*fixed.h5'
    reader = Reader(file_path, chunk_size=3)
    check_iterator_multi(reader.chunk_iterator())

    print("\nTest iteration over multiple files (w/ grouping):")
    file_path = r'..\test_data\test_data_set\[^.]*fixed.h5'
    reader = Reader(file_path, chunk_size=3, filegroup_groupby_pattern="_set\d+", filegroup_sortby_pattern="\d\w+.h5")
    check_iterator_multi(reader.chunk_iterator())

    print("\nTest iteration over multiple files (w/ grouping, w/ chunking):")
    file_path = r'..\test_data\test_data_set\[^.]*fixed.h5'
    reader = Reader(file_path, chunk_size=3, filegroup_groupby_pattern="_set\d+", filegroup_sortby_pattern="\d\w+.h5",
                    chunking_inmemory=True)
    check_iterator_multi(reader.chunk_iterator())

    print("\nTest iteration over multiple files (w/ file grouping, w/o chunking):")
    file_path = r'..\test_data\test_data_set\[^.]*fixed.h5'
    reader = Reader(file_path, chunk_size=-1, filegroup_groupby_pattern="_set\d+", filegroup_sortby_pattern="\d\w+.h5",
                    chunking_inmemory=True)
    check_iterator_multi(reader.chunk_iterator())

    print("\nTest iteration over multiple files (w/ file grouping, chunking by a specific column):")
    file_path = r'..\test_data\test_data_set\[^.]*fixed.h5'
    reader = Reader(file_path, chunk_by="group", filegroup_groupby_pattern="_set\d+",
                    filegroup_sortby_pattern="\d\w+.h5", chunking_inmemory=True)
    check_iterator_multi(reader.chunk_iterator())

    print("\nTest iteration over multiple files (w/o file grouping, chunking by a specific column):")
    file_path = r'..\test_data\test_data_set\[^.]*fixed.h5'
    reader = Reader(file_path, chunk_by="group", chunking_inmemory=True)
    check_iterator_multi(reader.chunk_iterator())

    print("\nTest pd.DataFrame object input ...")
    file_path = r'..\test_data\test_data_set\test_set1 - 1st_table.h5'
    file_path = pd.read_hdf(file_path)
    reader = Reader(file_path, chunk_size=2)
    check_iterator_multi(reader.chunk_iterator())
