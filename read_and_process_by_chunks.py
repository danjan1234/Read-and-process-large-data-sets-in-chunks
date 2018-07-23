"""
A simple workload scheduler that reads and processes large data files 
simultaneously by chunks

Author: Justin Duan
"""
import os
import time
import pandas as pd
import numpy as np
from multiprocessing import Process, Manager, cpu_count
from threading import Thread


def dummy_func(chunk_id, df):
    """
    Note:
        1.  The input arguments must be: chunk_id, pandas data frame object
        2.  The return value must be a dictionary with data frame name and data
            frame object pair. The data frame name is used to name the summary
            report to be saved
    """
    return {"dummy_result": pd.DataFrame(data=[1], columns=['a'])}


def count_rows(chunk_id, df):
    sum_df = pd.DataFrame.from_dict({'chunk_id': [chunk_id], 'chunk_size': [len(df)], 'min_row': [df.index.min()]})
    return {"summary_number_rows": sum_df}


class ReadProcessByChunks(object):
    """
    Read and process large data file by chunks. A simple workload scheduler is 
    implemented -- the main process reads the chunks into the buffer; simultaneously, 
    several child processes take the chunks out of the buffer and perform 
    user-defined data analysis. The results are collected from all child 
    processes to form a complete summary
    
    Attributes:
        self.head: read a few lines at the beginning (good for checking the number of columns or header strings)
        self.run: start the read-process run 
        self.save: save the summary files
    """
    STOP_FLAG = 'STOP'

    def __init__(self, file_path, binary=False, sep=',', header='infer', column_types=None,
                 chunk_size=1000, split_by=None, kwargs={},
                 n_buffered_chunks=10, n_jobs=None, func=None, func_kwargs={}, test_run=False):
        """
        Arguments:
            file_path: str, text or binary file
                File path or directory. Note if a directory is passed in, all files
                within this directory will be processed
            binary: bool, default=False
                Determine if the input file is binary
            sep: str, default ','
                Delimiter between the fields
            header: same as header argument in pd.read_csv
            column_types: dict or numpy.dtype, default None
                If a dictionary is passed, then it specifies the data types of
                columns of text files. This parameter is passed to dtype argument
                of pd.read_csv. Note, due to some limits in pandas implementation,
                one should avoid choosing 'category' to represent string columns,
                as this leads to concatenation errors. Instead 'object' data type
                should be specified
                if a numpy.dtype object is passed, then it assumes the input file
                is binary. It is used to parse each row of the binary file
            chunk_size: int
                The number of rows per chunk to read
            split_by: column name, default=None
                If specified, two read passes will be performed -- the first
                pass still reads the file in by input chunk size, the script
                creates a summary about the number of rows and order for each 
                unique split_by column values. After the first read pass finishes,
                an overall summary is created which serves as the guideline for
                the second read pass. This parameter ensures the complete set
                of data with the same split_by column values are read as one 
                single chunk 
            kwargs: dict, default {}
                Keyword argument to pass to pd.read_csv
            n_buffered_chunks: int
                The number of buffered chunks
            n_jobs: default None
                Number of CPU cores to use. Default None means number of cores - 1
            func: function object to apply to each chunk. Note func must only
                have two inputs: the chunk_id and chunk reference. The return 
                value of func must be a dictionary with keys being the dataframe
                names and values being the data frame objects
            func_kwargs: dict, default={}
                Keywords to pass to the processing function
            test_run: boolean, default False
                If True, then only the first chunk will be analyzed. This
                feature is good for debugging purpose
        """

        # Memorize the settings
        self._file_path = file_path
        self._binary = binary
        self._sep = sep
        self._header = header
        self._column_types = column_types
        self._chunk_size = chunk_size
        self._split_by = split_by
        self._n_buffered_chunks = n_buffered_chunks
        self._kwargs = kwargs
        self._n_jobs = cpu_count() - 1 if n_jobs is None else n_jobs
        self._func = func
        self._func_kwargs = func_kwargs
        self._test_run = test_run

    def head(self, n_lines=10):
        """
        Read a few lines from the target file. This function is supposed to be used separately

        :param
            n_lines: number of lines to read
        :return: None
        """
        with open(self._file_path) as f:
            lines = [f.readline() for _ in range(n_lines)]
        with open(self._file_path[:-4] + '_{}.{}'.format("head", self._file_path[-3:]), 'w') as f:
            f.writelines(lines)

    def _summarize_by_column(self, df):
        df.reset_index(inplace=True)
        df = pd.pivot_table(df, index=self._split_by, values='index',
                            aggfunc=(len, min))
        return df

    def _merge(self, dfs):
        if dfs is None or len(dfs) == 0:
            return

        return self._merge_helper(dfs, 0, len(dfs) - 1)

    def _merge_helper(self, dfs, start, end):
        if start == end:
            return dfs[start]

        mid = start + (end - start) // 2
        left = self._merge_helper(dfs, start, mid)
        right = self._merge_helper(dfs, mid + 1, end)
        return self._merge_two(left, right)

    @staticmethod
    def _merge_two(left, right):
        df = pd.merge(left, right, how='outer', left_index=True,
                      right_index=True, suffixes=('_l', '_r'))
        df[['len_l', 'len_r']] = df[['len_l', 'len_r']].fillna(0)
        df[['min_l', 'min_r']] = df[['min_l', 'min_r']].fillna(float('inf'))
        df['len'] = df['len_l'] + df['len_r']
        df['min'] = np.minimum(df['min_l'], df['min_r'])
        return df[['len', 'min']].astype(np.intc)

    def _create_chunks(self):
        if os.path.isdir(self._file_path):
            self._files = os.listdir(self._file_path)
        else:
            if self._binary:
                self._create_chunks_binary()
            else:
                self._create_chunks_text()

    def _create_chunks_binary(self):
        """
        Read the binary file. Note this function does not automatically split it to chunks
        """
        self._binary_file = open(self._file_path, 'rb')

    def _create_chunks_text(self):
        """
        Create chunks for the text file. Note if a split-by column is specified. The program will first count
        the number of rows corresponding to each entry in the split-by column. The correct number of rows will
        be read during the actual data processing step
        """
        self._chunks = pd.read_csv(self._file_path, chunksize=self._chunk_size,
                                   sep=self._sep, header=self._header,
                                   dtype=self._column_types, **self._kwargs)
        if self._split_by is not None:
            print("Build the summary based on the split-by column ...")
            dfs = []
            for chunk in self._chunks:
                df = self._summarize_by_column(chunk)
                dfs.append(df)
            self._split_bySumDf = self._merge(dfs)
            self._split_bySumDf.sort_values(by='min', inplace=True)
            self._reader = pd.read_csv(self._file_path, iterator=True,
                                       sep=self._sep, header=self._header,
                                       dtype=self._column_types, **self._kwargs)

    def _create_reader_thread(self):
        self._th = Thread(target=self._read_chunks_to_queue)
        self._th.start()

    def _read_chunks_to_queue(self):
        if os.path.isdir(self._file_path):
            self._read_single_file_to_queue()
        else:
            if self._binary:
                self._read_chunks_to_queue_binary()
            else:
                self._read_chunks_to_queue_text()

    def _read_single_file_to_queue(self):
        stop_flag = self.STOP_FLAG
        for chunk_id, file in enumerate(self._files):
            while self._chunk_queue.qsize() >= self._n_buffered_chunks:
                time.sleep(0.01)
            file_path = os.path.join(self._file_path, file)
            if self._binary:
                with open(file_path, 'rb') as f:
                    chunk = pd.DataFrame(np.fromfile(f, dtype=self._column_types))
                    self._chunk_queue.put((file, chunk))
            else:
                chunk = pd.read_csv(file_path, sep=self._sep, header=self._header, dtype=self._column_types, **self._kwargs)
                self._chunk_queue.put((file, chunk))    # Put file name in queue instead of chunk_id
            if self._test_run:
                break
        for _ in range(self._n_jobs):
            self._chunk_queue.put(stop_flag)

    def _read_chunks_to_queue_text(self):
        stop_flag = self.STOP_FLAG
        if self._split_by is None:
            for chunk_id, chunk in enumerate(self._chunks):
                while self._chunk_queue.qsize() >= self._n_buffered_chunks:
                    time.sleep(0.01)
                self._chunk_queue.put((chunk_id, chunk))
                if self._test_run:
                    break
        else:
            for chunk_id, chunk_size in enumerate(self._split_bySumDf['len']):
                while self._chunk_queue.qsize() >= self._n_buffered_chunks:
                    time.sleep(0.01)
                self._chunk_queue.put((chunk_id, self._reader.get_chunk(chunk_size)))
                if self._test_run:
                    break
        for _ in range(self._n_jobs):
            self._chunk_queue.put(stop_flag)

    def _read_chunks_to_queue_binary(self):
        stop_flag = self.STOP_FLAG
        chunk_id = 0
        while True:
            while self._chunk_queue.qsize() >= self._n_buffered_chunks:
                time.sleep(0.01)
            self._binary_file.seek(self._column_types.itemsize * self._chunk_size * chunk_id)
            chunk = pd.DataFrame(np.fromfile(self._binary_file, dtype=self._column_types, count=self._chunk_size))
            if len(chunk) <= 0:
                break
            self._chunk_queue.put((chunk_id, chunk))
            if self._test_run:
                break
            chunk_id += 1
        for _ in range(self._n_jobs):
            self._chunk_queue.put(stop_flag)

    def _create_queues(self):
        # Need to use multiprocessing.Manager, the Queue() is buggy and causes
        # deadlock
        manager = Manager()
        self._chunk_queue = manager.Queue()
        self._result_queue = manager.Queue()

    def _create_processes(self):
        self._processes = []
        for i in range(self._n_jobs):
            proc = Process(target=self._process_queue, daemon=False, kwargs={
                'process_id': i,
                'stop_flag': self.STOP_FLAG,
                'func': self._func,
                'func_kwargs': self._func_kwargs,
                'chunk_queue': self._chunk_queue,
                'result_queue': self._result_queue})
            self._processes.append(proc)
            proc.start()

    @classmethod
    def _process_queue(cls, process_id, stop_flag, func, func_kwargs, chunk_queue, result_queue):
        while True:
            if chunk_queue.qsize() <= 0:
                time.sleep(0.01)
                continue
            data = chunk_queue.get()
            if data == stop_flag:
                break
            chunk_id, chunk = data
            print("Processing chunk_id = {}".format(chunk_id))
            if func is None:
                continue
            data = func(chunk_id, chunk, **func_kwargs)
            if data is not None:
                result_queue.put(data)
        print("Worker {} is done!".format(process_id))

    def _wait_till_done(self):
        self._th.join()
        for proc in self._processes:
            if not proc.is_alive():
                continue
            proc.join()

    def _collect_results(self):
        self._wait_till_done()
        self._rlt = {}
        while self._result_queue.qsize() > 0:
            data = self._result_queue.get()
            if len(self._rlt) == 0:
                for key in data:
                    self._rlt[key] = [data[key]]
            else:
                for key in data:
                    self._rlt[key].append(data[key])

        for key in self._rlt:
            val = self._rlt[key]
            self._rlt[key] = pd.concat(val) if len(val) > 0 else pd.DataFrame()

        # Close file if necessary
        if not os.path.isdir(self._file_path):
            self._binary_file.close()

    def run(self):
        """
        Run the analysis

        :return: None
        """
        start_time = time.time()
        self._create_chunks()
        self._create_queues()
        self._create_reader_thread()
        self._create_processes()
        self._collect_results()
        print("Total data processing time: {:.0f} s".format(time.time() - start_time))

    def save(self, file_path, sep='\t', file_type='txt', post_process_func=None):
        """
        Arguments:
            file_path: file path to save the summary files
                Note suffixes will be added to the end of the filename to distinguish the
                purpose of each summary files
            file_type: the type of files to be saved. Options: 'txt', 'hdf'
            post_process_func: function to apply on the summary file
        """
        start_time = time.time()
        if not os.path.isdir(file_path):
            file_path = file_path[:-4]
        for i, key in enumerate(self._rlt):
            df = self._rlt[key]
            if post_process_func is not None:
                df = post_process_func(df)
            if file_type == 'txt':
                save_path = file_path + '_{}.txt'.format(key)
                df.to_csv(save_path, sep='\t', index=False)
            else:
                mode = 'w' if i == 0 else 'a'
                save_path = file_path + '_{}.h5'.format(key)
                df.to_hdf(save_path, key=key, format='fixed', mode=mode)
                # df.to_hdf(save_path, key=key, format='table', mode=mode, data_columns=True)
            print("Summary file is saved at: {}".format(save_path))
        print("Total data saving time: {:.0f} s".format(time.time() - start_time))


if __name__ == '__main__':
    # A test run
    file_path = r'C:\Users\justin.duan\Documents\Projects\Chip data\WER test 2\MPW8FastWER100K_180620_60_sites_2D_18addrs_1E-5.csv'
    sep, chunk_size = ',', 187200
    s = ReadProcessByChunks(file_path, sep=sep, split_by="CHIP",
                            chunk_size=chunk_size, header=2, n_buffered_chunks=20,
                            n_jobs=None, func=count_rows, test_run=False)
    s.run()
    s.save(file_path)
