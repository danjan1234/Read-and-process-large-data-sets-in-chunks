"""
A simple workload scheduler that reads and processes large data files 
simultaneously by chunks

Author: Justin Duan
"""

import time
import pandas as pd
from multiprocessing import Process, Manager, cpu_count
from threading import Thread


def dummyFunc(chunkID, df):
    """
    Note:
        1.  The input arguments must be: chunkID, pandas data frame object
        2.  The return value must be a dictionary with data frame name and data
            frame object pair. The data frame name is used to name the summary
            report to be saved
    """
    return {"dummy_result": pd.DataFrame()}
    
def countRows(chunkID, df):
    """
    Just count the number of rows in each chunk
    """
    sumDf = pd.DataFrame.from_dict({'chunk_size': [len(df)]})
    return {"summary_number_rows": sumDf}

    
class ReadAndProcessByChunks(object):
    """
    Read and process large data file by chunks. A simple workload scheduler is 
    implemented -- the main process reads the chunks into the buffer; simultaneously, 
    several child processes take the chunks out of the buffer and perform 
    user-defined data analysis. The results are collected from all child 
    processes to form a complete summary
    
    Attributes:
        self.run: start the read-process run 
        self.save: save the summary files
    """
    STOP_FLAG = 'STOP'
    
    def __init__(self, filePath, sep=',', columnTypes=None, chunkSize=1000, 
                    kwargs={}, nBufferedChunks=10, nJobs=None, func=None, 
                    testRun=False, warning=False):
        """
        Arguments:
            filePath: str
                The full path to the file to analysize
            sep: str, default ','
                Delimiter between the fields
            columnTypes: dict, default None
                A dictionary that specifies the data types of columns. This 
                parameter is passed to dtype argument of pd.read_csv. Note, due
                to some limits in pandas implementation, one should avoid 
                choosing 'category' to represent character data type, as this
                leads to concatenation errors. Instead 'object' data type should
                be specified
            chunkSize: int
                The number of rows per chunk to read
            kwargs: dict, default {}
                Keyword argument to pass to pd.read_csv
            nBufferedChunks: int
                The number of buffered chunks
            nJobs: default None
                Number of CPU cores to use. Default None means number of cores - 1
            func: function object to apply to each chunk. Note func must only
                have two inputs: the chunkID and chunk reference. The return 
                value of func must be a dictionary with keys being the dataframe
                names and values being the data frame objects
            testRun: boolean, default False
                If True, then only the first chunk will be analysized. This
                feature is good for debugging purpose
                
        """
            
        # Memorize the settings
        self._filePath = filePath
        self._sep = sep
        self._columnTypes = columnTypes
        self._chunkSize = chunkSize
        self._nBufferedChunks = nBufferedChunks
        self._kwargs = kwargs
        self._nJobs = cpu_count() - 1 if nJobs is None else nJobs
        self._func = func
        self._testRun = testRun
    
    def _createChunks(self):
        self._chunks = pd.read_csv(self._filePath, chunksize=self._chunkSize,
                        sep=self._sep, dtype=self._columnTypes, **self._kwargs)
    
    def _createReaderThread(self):
        self._th = Thread(target=self._readChunksToQueue)
        self._th.start()
    
    def _readChunksToQueue(self):
        stopFlag = self.STOP_FLAG
        for chunkID, chunk in enumerate(self._chunks):
            while self._chunkQueue.qsize() >= self._nBufferedChunks:
                time.sleep(0.01)
            self._chunkQueue.put((chunkID, chunk))
            if self._testRun:
                break
        for _ in range(self._nJobs):
            self._chunkQueue.put(stopFlag)
                            
    def _createQueues(self):
        # Need to use multiprocessing.Manager, the Queue() is buggy and causes
        # deadlock
        manager = Manager()
        self._chunkQueue = manager.Queue()
        self._resultQueue = manager.Queue()
    
    def _createProcesses(self):
        self._processes = []
        for i in range(self._nJobs):
            proc = Process(target=self._processQueue, daemon=False, kwargs={
                            'processID': i,
                            'stopFlag': self.STOP_FLAG,
                            'func': self._func,
                            'chunkQueue': self._chunkQueue,
                            'resultQueue': self._resultQueue})
            self._processes.append(proc)
            proc.start()
            
    @classmethod     
    def _processQueue(cls, processID, stopFlag, func, chunkQueue, resultQueue):
        while True:
            if chunkQueue.qsize() <= 0:
                time.sleep(0.01)
                continue
            data = chunkQueue.get()
            if data == stopFlag:
                break
            chunkID, chunk = data
            print("Processing chunkID = {}".format(chunkID))
            if func is None:
                continue
            data = func(chunkID, chunk)
            if data is not None:
                resultQueue.put(data)
        print("Worker {} is done!".format(processID))
    
    def _waitTillDone(self):
        self._th.join()
        for proc in self._processes:
            if not proc.is_alive():
                continue
            proc.join()
        
    def _collectResults(self):
        self._waitTillDone()
        self._rlt = {}
        while self._resultQueue.qsize() > 0:
            data = self._resultQueue.get()
            if len(self._rlt) == 0:
                for key in data:
                    self._rlt[key] = [data[key]]
            else:
                for key in data:
                    self._rlt[key].append(data[key])
       
        for key in self._rlt:
            val = self._rlt[key]
            self._rlt[key] = pd.concat(val) if len(val) > 0 else pd.DataFrame()
     
    def run(self):
        _startTime = time.time()
        self._createChunks()
        self._createQueues()
        self._createReaderThread()
        self._createProcesses()
        self._collectResults()
        print("Total runtime: {:.0f} s".format(time.time() - _startTime))
    
    def save(self, filePath, sep='\t'):
        """
        Arguments:
            filePath: file path to save the summary files
        """
        for key in self._rlt:
            df = self._rlt[key]
            df.to_csv(filePath[:-4] + '_{}.txt'.format(key), sep='\t')
 
if __name__ == '__main__':
    pass