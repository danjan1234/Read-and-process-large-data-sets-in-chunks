"""
A simple workload scheduler that reads and processes large data files 
simultaneously by chunks

Author: Justin Duan
Time: 2018/07/26 4:22PM
"""
import os
import glob
import time
import pandas as pd
import numpy as np
from multiprocessing import Process, Manager, cpu_count
from threading import Thread


def dummy_func(id, df):
    """
    Note:
        1.  The input arguments must be: id, pandas dataframe object
        2.  The return value must be a dictionary with string and dataframe pair. The string (key) is used to name the
            dataframe during save
    """
    return {"dummy_result": pd.DataFrame(data=[1], columns=['a'])}


def count_rows(id, df):
    """
    Just count the number of rows
    """
    sum_df = pd.DataFrame.from_dict({'id': [id], 'chunk_size': [len(df)], 'min_row': [df.index.min()]})
    return {"summary_number_rows": sum_df}


class ReadProcessByChunks(object):
    """
    Read and process large data file by chunks. A simple workload scheduler is implemented -- the main process reads
    the chunks into the buffer; simultaneously, several child processes take the chunks out of the buffer and perform
    user-defined data analysis. The results are collected from all child processes to form a complete summary
    
    Attributes:
        self.head: read a few lines at the beginning (good for checking the number of columns or header strings)
        self.run: start the read-process run 
        self.save: save the summary files
    """
    STOP_FLAG = 'STOP'

    def __init__(self, file_path, dtype=None, chunk_size=1000, split_by=None, kwargs={}, n_buffered_chunks=-1,
                 func=None, func_kwargs={}, keep_rlt=False, post_process_func=None, post_func_kwargs={},
                 save=True, save_path=None, n_jobs=-1, test_run=False):
        """
        Arguments:
            file_path: str, text or binary file
                File path. Multiple files are supported with * wild card. Note if * is used, all files in the directory
                with names matching the pattern will be processed
            dtype: dict or numpy.dtype, default None
                If a dictionary is passed, then it specifies the data types of columns of text files. This parameter
                is passed to dtype argument of pd.read_csv. Note, due to some limits in pandas implementation, one
                should avoid choosing 'category' to represent string columns, as this leads to concatenation errors.
                Instead 'object' data type should be specified if a numpy.dtype object is passed, then it assumes the
                input file is binary. It is used to parse each row of the binary file
            chunk_size: int
                The number of rows per chunk to read
            split_by: column name, default=None
                If specified, two read passes will be performed -- the first pass still reads the file in by the
                specified chunk size, the program creates a summary based on this column. It then knows how many rows
                should be read for each unique value of the split_by column. Note the split_by column selected must
                have its value sorted in the entire data set. This parameter ensures the read does not break within the
                section with the same split_by value
            kwargs: dict, default {}
                Keyword argument to pass to file reading functions
            n_buffered_chunks: int, default=-1
                The number of buffered chunks. If -1, then a value equals to 2 * n_jobs  is used
            func: function object to apply to each chunk
                Note func must only have two inputs: the id and chunk reference. The return value of func must be a
                dictionary with keys being the dataframe names and values being the data frame objects
            func_kwargs: dict, default={}
                Keywords to pass to the processing function
            keep_rlt: bool, default=False
                Whether or not the final result should be kept in the memory. It's convenient if the result needs to be
                passed to another processing routine. Note: please watch the memory usage by yourself
            post_process_func: function object to apply to the final result
                Note if this parameter is not None, the final result will be preserved in the memory regardless of
                keep_rlt setting. If it makes no difference applying the post process function to each chunk than the
                entire final data set, it is advised to do it on each chunk. This avoids the need to preserving the
                result in the memory
            post_func_kwargs: dict, default={}
                Keywords to pass to the post processing function
            save: bool, default=True
                Save the summary results
            save_path: str, default=None
                File or directory path to save the final summary. If None, then file_path will be used
            n_jobs: int, default=-1
                Number of CPU cores to use. If -1, the value is set to number of cores - 1
            test_run: boolean, default False
                If True, then only the first chunk will be analyzed. This
                feature is good for debugging purpose
        """
        self._start_time = time.time()

        # Memorize the settings
        self._file_path = file_path
        self._dtype = dtype
        self._chunk_size = chunk_size
        self._split_by = split_by
        self._kwargs = kwargs
        self._n_jobs = cpu_count() - 1 if n_jobs == -1 else n_jobs
        self._n_buffered_chunks = 2 * self._n_jobs if n_buffered_chunks == -1 else n_buffered_chunks
        self._func = func
        self._func_kwargs = func_kwargs
        self._keep_rlt = keep_rlt
        self._post_process_func = post_process_func
        self._post_func_kwargs = post_func_kwargs
        self._test_run = test_run
        self._save = save
        self._save_path = save_path if save_path is not None else file_path
        self._rlt = {}
        self._binary_file = None

        if not os.path.exists(self._save_path):
            os.makedirs(self._save_path)

    def head(self, n_lines=10):
        """
        Read a few lines from the target file. Note directory is not supported. This function should be used separately
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
        # Multiple files
        if '*' in self._file_path:
            directory, file_name = os.path.split(self._file_path)
            self._files = glob.glob(os.path.join(directory, "**", file_name), recursive=True)
        # Single files
        else:
            if self._file_path[-4:] == '.bin':
                self._create_chunks_binary()
            elif self._file_path[-4:] in {'.txt', '.csv'}:
                self._create_chunks_non_binary()
            elif self._file_path[-3:] == '.h5':
                self._create_chunks_non_binary()

    def _create_chunks_binary(self):
        """
        Read the binary file. Note this function does not automatically split it to chunks
        """
        self._binary_file = open(self._file_path, 'rb')

    def _create_chunks_non_binary(self):
        """
        Create chunks for the text file. Note if a split-by column is specified. The program will first count
        the number of rows corresponding to each entry in the split-by column. The correct number of rows will
        be read during the actual data processing step
        """
        if self._file_path[-4:] in {'.txt', '.csv'}:
            self._chunks = pd.read_csv(self._file_path, chunksize=self._chunk_size, dtype=self._dtype, **self._kwargs)
        elif self._file_path[-3:] == '.h5':
            self._chunks = pd.read_hdf(self._file_path, chunksize=self._chunk_size, **self._kwargs)
        if self._split_by is not None:
            print("Build the summary based on the split-by column ...")
            dfs = []
            for chunk in self._chunks:
                df = self._summarize_by_column(chunk)
                dfs.append(df)
            self._split_bySumDf = self._merge(dfs)
            self._split_bySumDf.sort_values(by='min', inplace=True)
            if self._file_path[-4:] in {'.txt', '.csv'}:
                self._reader = pd.read_csv(self._file_path, iterator=True, dtype=self._dtype, **self._kwargs)
            elif self._file_path[-3:] == '.h5':
                self._reader = pd.read_hdf(self._file_path, iterator=True, **self._kwargs)

    def _create_reader_thread(self):
        self._th_reader = Thread(target=self._read_chunks_to_queue)
        self._th_reader.start()
        
    def _read_chunks_to_queue(self):
        if '*' in self._file_path:
            self._read_single_file_to_queue()
        else:
            if self._file_path[-4:] == '.bin':
                self._read_chunks_to_queue_binary()
            elif self._file_path[-4:] in {'.txt', '.csv'}:
                self.read_chunks_to_queue_non_binary()
            elif self._file_path[-3:] == '.h5':
                self.read_chunks_to_queue_non_binary()
        # Signal the workers to stop
        for _ in range(self._n_jobs):
            self._chunk_queue.put(self.STOP_FLAG)

    def _read_single_file_to_queue(self):
        for id, file_path in enumerate(self._files):
            _, file_name = os.path.split(file_path)
            id = "{}, {}".format(id, file_name)
            while self._chunk_queue.qsize() >= self._n_buffered_chunks:
                time.sleep(0.01)
            if self._file_path[-4:] == '.bin':
                with open(file_path, 'rb') as f:
                    chunk = pd.DataFrame(np.fromfile(f, dtype=self._dtype))
                    self._chunk_queue.put((id, chunk))
            elif self._file_path[-4:] in {'.txt', '.csv'}:
                chunk = pd.read_csv(file_path, dtype=self._dtype, **self._kwargs)
                self._chunk_queue.put((id, chunk))
            elif self._file_path[-3:] == '.h5':
                chunk = pd.read_hdf(file_path, **self._kwargs)
                self._chunk_queue.put((id, chunk))
            if self._test_run:
                break

    def read_chunks_to_queue_non_binary(self):
        if self._split_by is None:
            for id, chunk in enumerate(self._chunks):
                while self._chunk_queue.qsize() >= self._n_buffered_chunks:
                    time.sleep(0.01)
                self._chunk_queue.put((id, chunk))
                if self._test_run:
                    break
        else:
            for id, chunk_size in enumerate(self._split_bySumDf['len']):
                while self._chunk_queue.qsize() >= self._n_buffered_chunks:
                    time.sleep(0.01)
                self._chunk_queue.put((id, self._reader.get_chunk(chunk_size)))
                if self._test_run:
                    break

    def _read_chunks_to_queue_binary(self):
        id = 0
        while True:
            while self._chunk_queue.qsize() >= self._n_buffered_chunks:
                time.sleep(0.01)
            self._binary_file.seek(self._dtype.itemsize * self._chunk_size * id)
            chunk = pd.DataFrame(np.fromfile(self._binary_file, dtype=self._dtype, count=self._chunk_size))
            if len(chunk) <= 0:
                break
            self._chunk_queue.put((id, chunk))
            if self._test_run:
                break
            id += 1

    def _create_queues(self):
        # Need to use multiprocessing.Manager, the Queue() is buggy and causes
        # deadlock
        manager = Manager()
        self._chunk_queue = manager.Queue()
        self._result_queue = manager.Queue()
        self._log_queue = manager.Queue()

    def _create_processes(self):
        self._processes = []
        for i in range(self._n_jobs):
            proc = Process(target=self._process_queue, kwargs={
                'process_id': i,
                'stop_flag': self.STOP_FLAG,
                'func': self._func,
                'func_kwargs': self._func_kwargs,
                'chunk_queue': self._chunk_queue,
                'result_queue': self._result_queue,
                'log_queue': self._log_queue})
            self._processes.append(proc)
            proc.start()

    @classmethod
    def _process_queue(cls, process_id, stop_flag, func, func_kwargs, chunk_queue, result_queue, log_queue):
        while True:
            if chunk_queue.qsize() <= 0:
                time.sleep(0.01)
                continue
            print("Number of chunks in memory: {}".format(chunk_queue.qsize()))
            data = chunk_queue.get()
            if data == stop_flag:
                break
            id, chunk = data
            print("Processing id = {}".format(id))
            if func is None:
                continue
            try:
                data = func(id, chunk, **func_kwargs)
            except Exception as e:
                error_msg = ("Error has occurred to chunk id = {}. Process continues to the next chunk. "
                             "Error message: {}".format(id, e))
                print(error_msg)
                log_queue.put(error_msg)
                data = None
            if data is not None:
                result_queue.put(data)
        print("Worker {} is done!".format(process_id))

    def _create_saver_thread(self):
        self._th_saver = Thread(target=self._save_queue_to_disk)
        self._th_saver.start()

    def _save_queue_to_disk(self):
        """
        Save the final results to the disk. Note there are two approaches to save the results, depending on whether or
        not the post processing function is applied or not. Only hdf file format is supported
        """
        save_path = self._save_path
        n_proc_alive = float('inf')
        just_started = True
        mode = 'w'
        initial_write = True
        # Keep working if there're any live data analysis processes or any unsaved items in the queue
        while just_started or n_proc_alive > 0 or self._result_queue.qsize() > 0:
            time.sleep(0.01)
            n_proc_alive = 0
            for proc in self._processes:
                if proc.is_alive():
                    n_proc_alive += 1
                    just_started = False
            if self._result_queue.qsize() > 0:
                data = self._result_queue.get()
                if self._save and self._post_process_func is None:
                    for key in data:
                        if len(data[key]):
                            file_path = os.path.join(save_path, '{}.h5'.format(key))
                            if os.path.isfile(file_path) and not initial_write:
                                mode = 'a'
                            data[key].to_hdf(file_path, key='summary', format='table', mode=mode, data_columns=True,
                                             append=True)
                            initial_write = False
                            print("Attn: summary file is saved at: {}".format(file_path))
                if self._keep_rlt or (self._save and self._post_process_func is not None):
                    if len(self._rlt) == 0:
                        for key in data:
                            self._rlt[key] = []
                    for key in data:
                        if len(data[key]):
                            self._rlt[key].append(data[key])

        # Save after applying post function
        if self._keep_rlt or (self._save and self._post_process_func is not None):
            for key in self._rlt:
                self._rlt[key] = pd.concat(self._rlt[key]) if len(self._rlt[key]) > 0 else pd.DataFrame()
                if self._post_process_func is not None:
                    self._rlt[key] = self._post_process_func(self._rlt[key], **self._post_func_kwargs)
                if self._save:
                    if len(self._rlt[key]):
                        file_path = os.path.join(save_path, '{}.h5'.format(key))
                        self._rlt[key].to_hdf(file_path, key='summary', format='table', mode='w', data_columns=True)
                        print("Attn: summary file is saved at: {}".format(file_path))

    def _close(self):
        # Close file if necessary
        self._th_saver.join()
        if self._binary_file is not None:
            self._binary_file.close()

        # Print the total run time
        msg = "Done! Total processing time: {:.0f} s".format(time.time() - self._start_time)
        print(msg)
        self._log_queue.put(msg)

        # Save the log
        log = []
        while self._log_queue.qsize() > 0:
            log.append(self._log_queue.get() + '\n')
        with open(os.path.join(self._save_path, "log.txt"), 'w') as log_f:
            log_f.writelines(log)

    def run(self):
        """
        Run the analysis
        """
        self._create_chunks()
        self._create_queues()
        self._create_reader_thread()
        self._create_processes()
        self._create_saver_thread()
        self._close()


if __name__ == '__main__':
    # A test run
    file_path = r'tmp.csv'
    s = ReadProcessByChunks(file_path)
    s.run()
