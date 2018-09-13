"""
A simple workload scheduler that reads and processes large data files simultaneously by chunks

Author: Justin Duan
Time: 2018/09/13 12:05PM

Object design:
    Log: takes care of log file writing, log queue maintenance, as wall as message printing

    Reader: takes care of various file input scenarios. No matter how complicated the situation is, it supplies a
    chunk iterator that outputs one chunk at a time

    ReadProcessByChunks: main class
"""

import shutil
import os
import time
import pandas as pd
from multiprocessing import Process, Manager, cpu_count
from threading import Thread
from reader import Reader
from log import Log


# Example: function used to count the number of rows
def count_rows(df_tuple_list, col_names=('len', 'min_row', 'input_file_path')):
    """
    Note: both the input and output must be a list of tuples and the elements within each tuple must be: file_path,
    chunk_id, DataFrame. Extra input arguments are allowed

    The consistent input and output format allows function chaining
    """
    dfs = []
    overall_path = []
    for file_path, chunk_id, df in df_tuple_list:
        overall_path.append(os.path.split(file_path)[1])
        tmp_df = pd.DataFrame.from_dict({col_names[0]: len(df), col_names[1]: [df.index.min()],
                                         col_names[2]: file_path})
        dfs.append(tmp_df)

    return [(" & ".join(overall_path), df_tuple_list[0][1], pd.concat(dfs))]


class ReadProcessByChunks(object):
    """
    Read and process large data file by chunks. A simple workload scheduler is implemented -- the main process reads
    the chunks into the buffer queue; simultaneously, several child processes take the chunks out of the buffer and
    perform user-defined data analysis. The results are collected from all child processes and form a complete summary
    
    Attributes:
        self.run: start the read-process run
    """
    STOP_FLAG = 'STOP'

    def __init__(self, file_path, save_dir_path, purge_save_directory=False, save_mode=0, delayed_save=False,
                 hdf_format='table', filegroup_groupby_pattern=None, filegroup_sortby_pattern=None, dtype=None,
                 chunk_size=-1, chunk_by=None, chunking_inmemory=False, kwargs={}, n_buffered_chunks=-1,
                 n_buffered_results=-1, func=None, func_kwargs={}, n_jobs=-1, chunk_limit=-1, sum_prefix='sum'):
        """
        Arguments:
            file_path: str
                File path(s). Multiple files are supported with * wild card. Note if * is used, all files in the
                corresponding directory that matches the pattern will be processed
            save_dir_path: str, default=None
                Directory path to save the final summaries
            purge_save_directory: bool, default=False
                Whether or not the save directory should be deleted and recreated before saving the new results
            save_mode: int, default=0
                Determine how the final summary data should be saved. 0: by file and chunk; 1: by file only; 2: by
                None (all chunks from different files will be saved to one file)
            delayed_save: bool, default=False
                Determine whether or not the data should be preserved in memory and saved altogether before program
                terminates. Using this option can improve the file saving efficiency especially when the individual
                files to save are small. This option also removes the limit to save in 'fixed' hdf format since the
                file is only saved once
            hdf_format: str, options={'fixed', 'table'}, default='fixed'
                This setting affects pd.to_hdf's table save format. Note 'fixed' does not support appending, hence
                save_mode = 0 is always suggested with 'fixed' format is selected
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
                If true, the file(s) will be read into memory as a whole and split into chunks afterward
            kwargs: dict, default={}
                Keyword argument passed to file reading functions (pd.read_csv, pd.read_hdf, np.fromfile, etc.)
            n_buffered_chunks: int, default=-1
                The number of buffered chunks in memory. If -1, then a value equals to 2 * n_jobs  is used
            n_buffered_results: int, default=-1
                The number of buffered results in memory. If -1, then a value equals to 2 * n_jobs  is used
            func: function object to apply to each chunk
                Note func must only have two inputs: the id and chunk reference. The return value of func must be a
                dictionary with keys being the DataFrame names and values being the data frame objects
            func_kwargs: dict, default={}
                Keywords to pass to the processing function
            n_jobs: int, default=-1
                Number of CPU cores to use. If -1, the value is set to number of cores - 1
            chunk_limit: uint, default -1
                The maximum number of chunks to process. If -1, then all chunks will be processed. It's convenient for
                debugging purpose
            :sum_prefix: str, default='sum_'. Prefix string for the summary files
        """
        self._start_time = time.time()

        # Memorize the settings
        self._file_path = file_path
        self._save_mode = save_mode
        self._hdf_format = hdf_format
        self._delayed_save = delayed_save
        self._filegroup_groupby_pattern = filegroup_groupby_pattern
        self._filegroup_sortby_pattern = filegroup_sortby_pattern
        self._dtype = dtype
        self._chunk_size = chunk_size
        self._chunk_by = chunk_by
        self._chunking_inmemory = chunking_inmemory
        self._kwargs = kwargs
        self._n_jobs = cpu_count() - 1 if n_jobs == -1 else n_jobs
        self._n_buffered_chunks = 2 * self._n_jobs if n_buffered_chunks == -1 else n_buffered_chunks
        self._n_buffered_results = 2 * self._n_jobs if n_buffered_results == -1 else n_buffered_results
        self._func = func
        self._func_kwargs = func_kwargs
        self._chunk_limit = chunk_limit
        self._save_dir_path = save_dir_path
        self._sum_prefix = sum_prefix

        # Create the log object
        self._log = Log(os.path.join(self._save_dir_path, 'log.txt'), log_interval=10)

        if not os.path.exists(self._save_dir_path):
            os.makedirs(self._save_dir_path)
        else:
            if purge_save_directory:
                shutil.rmtree(self._save_dir_path)
                time.sleep(1)
                os.makedirs(self._save_dir_path)

    def _create_reader_thread(self):
        self._th_reader = Thread(target=self._read_chunks_to_queue)
        self._th_reader.start()
        
    def _read_chunks_to_queue(self):
        chunk_it = Reader(self._file_path, self._filegroup_groupby_pattern, self._filegroup_sortby_pattern, 
                          self._dtype, self._chunk_size, self._chunk_by, self._chunking_inmemory, self._kwargs,
                          self._log).chunk_iterator()
        try:
            chunk_id = 0
            while True:
                # Wait for the chunk queue to dissipate
                while self._chunk_queue.qsize() >= self._n_buffered_chunks:
                    time.sleep(0.01)
                self._chunk_queue.put(next(chunk_it))
                chunk_id += 1
                if chunk_id >= self._chunk_limit > 0:
                    raise StopIteration
        except StopIteration:
            msg = "Reached the end of chunks"
            self._log.put(msg)
        
        # Signal the workers to stop
        for _ in range(self._n_jobs):
            self._chunk_queue.put(self.STOP_FLAG)
            
    def _create_queues(self):
        # Need to use multiprocessing.Manager, the Queue() is buggy and causes
        # deadlock
        manager = Manager()
        self._chunk_queue = manager.Queue()
        self._result_queue = manager.Queue()

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
                'n_buffered_results': self._n_buffered_results,
                'log': self._log})
            self._processes.append(proc)
            proc.start()

    @classmethod
    def _process_queue(cls, process_id, stop_flag, func, func_kwargs, chunk_queue, result_queue, n_buffered_results, 
                       log):
        while True:
            if chunk_queue.qsize() <= 0 or result_queue.qsize() > n_buffered_results:
                time.sleep(0.01)
                continue
            data = chunk_queue.get()
            if data == stop_flag:
                break
            msg = "Processing filename (1st)='{}', ch={}. # of data and result chunks in memory: {}, {}".format(
                data[0][0], data[0][1], chunk_queue.qsize(), result_queue.qsize())
            log.put(msg)
            if func is None:
                continue
            try:
                result = func(data, **func_kwargs)
                result_queue.put(result)
            except Exception as e:
                error_msg = ("Error has occurred to filename (1st)='{}', ch={}. Process continues to the next chunk. "
                             "Error message: {}".format(data[0][0], data[0][1], e))
                log.put(error_msg)
        msg = "Worker {} is done!".format(process_id)
        log.put(msg)

    def _create_saver_thread(self):
        self._th_saver = Thread(target=self._save_queue_to_disk)
        self._th_saver.start()

    def _save_queue_to_disk(self):
        """
        Save the final results to the disk. Note there are two approaches to save the results, depending on whether or
        not the post processing function is applied or not. Only hdf file format is supported
        """
        if self._save_dir_path is None:
            return
        n_proc_alive = float('inf')
        just_started = True
        df_dic = {}
        mode = 'w'

        # Keep working if there're any live data analysis processes or any unsaved items in the queue
        while just_started or n_proc_alive > 0 or self._result_queue.qsize() > 0:
            time.sleep(0.01)
            n_proc_alive = 0
            for proc in self._processes:
                if proc.is_alive():
                    n_proc_alive += 1
                    just_started = False

            # Write logs
            self._log.write()

            # Write results
            if self._result_queue.qsize() > 0:
                queue_data = self._result_queue.get()
                if queue_data is None:
                    continue
                for summary_id, data in enumerate(queue_data):
                    if data is None:
                        continue
                    file_path, chunk_id, df = data
                    if len(df) == 0:
                        continue
                    hdf_kwargs = {}
                    file_save_path = os.path.join(self._save_dir_path, file_path)
                    if self._save_mode == 0:
                        file_save_path += ', ch={}, {}_id={}.h5'.format(chunk_id, self._sum_prefix, summary_id)
                    elif self._save_mode == 1:
                        file_save_path += ', {}_id={}.h5'.format(self._sum_prefix, summary_id)
                        df['_chunk_id'] = chunk_id
                    else:
                        file_save_path = os.path.join(self._save_dir_path, '{}_id={}.h5'.format(self._sum_prefix,
                                                                                                summary_id))
                        df['_file_path'] = file_path
                        df['_chunk_id'] = chunk_id
                        hdf_kwargs['min_itemsize'] = {'_file_path': 128}
                    if self._delayed_save:
                        if file_save_path not in df_dic:
                            df_dic[file_save_path] = []
                        df_dic[file_save_path].append(df)
                    else:
                        self._save(file_save_path, df, hdf_kwargs, mode=mode)
                    self._log.put("Attn: summary file is saved at: {}".format(file_save_path))
                mode = 'a'

        # Delayed save
        for file_save_path, dfs in df_dic.items():
            df = pd.concat(dfs)
            self._save(file_save_path, df, mode='w')

    def _save(self, file_save_path, df, hdf_kwargs={}, mode='a'):
        """
        Save an individual file
        """
        self._create_directories(file_save_path)
        append = True if self._hdf_format == 'table' else False
        df.to_hdf(file_save_path, key='summary', format=self._hdf_format, mode=mode, data_columns=True, append=append,
                  **hdf_kwargs)

    def _create_directories(self, file_save_path):
        """
        Create all required parent directories associated with file_save_path
        """
        dir_path = os.path.split(file_save_path)[0]
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def _close(self):
        # Close file if necessary
        self._th_saver.join()

        # Print the total run time
        msg = "Done! Total processing time: {:.0f} s".format(time.time() - self._start_time)
        self._log.put(msg)

        # Save the log
        self._log.write(log_interval=0)

    def run(self):
        """
        Run the analysis
        """
        self._create_queues()
        self._create_reader_thread()
        self._create_processes()
        self._create_saver_thread()
        self._close()


if __name__ == '__main__':
    # A few test runs
    import shutil

    # print("\nSingle file read as a whole:")
    # file_path = r'C:\Users\justin.duan\Documents\Projects\Chip data\Chip_test_scripts - dev\test_data_set\test_set1 - 1st_fixed.h5'
    # save_dir_path = r'C:\Users\justin.duan\Documents\Projects\Chip data\Chip_test_scripts - dev\test_data_set_summary'
    # shutil.rmtree(save_dir_path)
    # s = ReadProcessByChunks(file_path, save_dir_path, func=count_rows, hdf_format='table')
    # s.run()

    # print("\nSingle file by chunk size:")
    # file_path = r'C:\Users\justin.duan\Documents\Projects\Chip data\Chip_test_scripts - dev\test_data_set\test_set1 - 1st_fixed.h5'
    # save_dir_path = r'C:\Users\justin.duan\Documents\Projects\Chip data\Chip_test_scripts - dev\test_data_set_summary'
    # shutil.rmtree(save_dir_path)
    # s = ReadProcessByChunks(file_path, save_dir_path, chunk_size=3, func=count_rows, hdf_format='table')
    # s.run()

    # print("\nMultiple files (each file is read in in intact):")
    # file_path = r'C:\Users\justin.duan\Documents\Projects\Chip data\Chip_test_scripts - dev\test_data_set\*fixed.h5'
    # save_dir_path = r'C:\Users\justin.duan\Documents\Projects\Chip data\Chip_test_scripts - dev\test_data_set_summary'
    # shutil.rmtree(save_dir_path)
    # s = ReadProcessByChunks(file_path, save_dir_path, func=count_rows, hdf_format='table')
    # s.run()

    # print("\nMultiple file (each file is broken to chunks of fixed sizes):")
    # file_path = r'C:\Users\justin.duan\Documents\Projects\Chip data\Chip_test_scripts - dev\test_data_set\*fixed.h5'
    # save_dir_path = r'C:\Users\justin.duan\Documents\Projects\Chip data\Chip_test_scripts - dev\test_data_set_summary'
    # shutil.rmtree(save_dir_path)
    # s = ReadProcessByChunks(file_path, save_dir_path, chunk_size=3, func=count_rows, hdf_format='table')
    # s.run()

    # print("\nMultiple file (each file is broken to chunks by a certain column):")
    # file_path = r'C:\Users\justin.duan\Documents\Projects\Chip data\Chip_test_scripts - dev\test_data_set\*fixed.h5'
    # save_dir_path = r'C:\Users\justin.duan\Documents\Projects\Chip data\Chip_test_scripts - dev\test_data_set_summary'
    # shutil.rmtree(save_dir_path)
    # s = ReadProcessByChunks(file_path, save_dir_path, chunk_by='group', func=count_rows, hdf_format='table')
    # s.run()

    # print("\nMultiple file (with grouping, each file is broken to chunks by a certain column):")
    # file_path = r'C:\Users\justin.duan\Documents\Projects\Chip data\Chip_test_scripts - dev\test_data_set\*fixed.h5'
    # save_dir_path = r'C:\Users\justin.duan\Documents\Projects\Chip data\Chip_test_scripts - dev\test_data_set_summary'
    # shutil.rmtree(save_dir_path)
    # s = ReadProcessByChunks(file_path, save_dir_path, filegroup_groupby_pattern="_set\d+",
    #                         filegroup_sortby_pattern="\d\w+.h5", chunk_by='group', func=count_rows, hdf_format='table')
    # s.run()

    # print("\nTest save by file only:")
    # file_path = r'C:\Users\justin.duan\Documents\Projects\Chip data\Chip_test_scripts - dev\test_data_set\*fixed.h5'
    # save_dir_path = r'C:\Users\justin.duan\Documents\Projects\Chip data\Chip_test_scripts - dev\test_data_set_summary'
    # shutil.rmtree(save_dir_path)
    # s = ReadProcessByChunks(file_path, save_dir_path, save_mode=1, filegroup_groupby_pattern="_set\d+",
    #                         filegroup_sortby_pattern="\d\w+.h5", chunk_by='group', func=count_rows, hdf_format='table')
    # s.run()

    print("\nTest save by none (save everything in one file):")
    file_path = r'C:\Users\justin.duan\Documents\Projects\Chip data\Chip_test_scripts - dev\test_data_set\*fixed.h5'
    save_dir_path = r'C:\Users\justin.duan\Documents\Projects\Chip data\Chip_test_scripts - dev\test_data_set_summary'
    shutil.rmtree(save_dir_path)
    s = ReadProcessByChunks(file_path, save_dir_path, save_mode=2, filegroup_groupby_pattern="_set\d+",
                            filegroup_sortby_pattern="\d\w+.h5", chunk_by='group', func=count_rows, hdf_format='table')
    s.run()
