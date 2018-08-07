# Read and process large data sets in chunks using multiple cores in single machines

## What does this module do
This compact Python module creates a simple task manager for reading and processing large data sets in chunks. The data file is split to multiple chunks and a number of them are read into the memory initially. The task manager then distributes the read chunks to multiple cores for processing. When the number of remaining unprocessed chunks in the memory drops below some user-defined threshold, new chunks are read and the process continues

## How to use
You need to supply your own data analysis routine. In order to be compatible with the task manager, some rules must be followed. First of all, the argument list of the custom analysis routine must have two inputs: chunkID and df – the former is the rank of the chunk passed in and the latter is a local Pandas DataFrame reference to this chunk. One should not be too concerned about the magic here as long as neither input is forgotten. Secondly, in order for the task manager to collect the processed results, the return type must be in the form of a dictionary with keys being the names of the resultant Pandas DataFrames and the values being these DataFrame objects. Note the keys will be used to name these DataFrames when saving. As an example, I have included two dummy routines (“dummyFunc” and “countRows”) 

## What is this module good for and what is it not good for
This module is especially good for processing large data sets resided in a single machine utilizing all cores. There is no limit in the size of the original data sets as long as the processed result does not blow up the memory

## Why not Dask?
First of all, this program is light weighted and only depends on Pandas and Multiprocessing modules, while Dask still demands some effort for one to be familiar with. Secondly, each chunk is a legitimate Pandas DataFrame so there is no compatibility issue with Pandas, however, as for Dask, not all Pandas operations are supported

## Under the hood
The task manager has one process dedicated to reading and distributing chunks and several other processes for data analysis. The main process reads the chunks into a shared queue until the total number of chunks in the queue reaches a used-defined limit. The other worker processes remove the chunks from the queue and perform required analysis. The processed results are stored in a second queue dedicated for data collection. Once all chunks are processed, all worker processes are terminated and the main process collects the results from the second queue and generate a complete summary
