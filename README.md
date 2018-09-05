# Read and process large data sets in chunks using multiple cores in single machines

## What does this module do?
This compact Python module creates a simple task manager for reading and processing large data sets in chunks. The following scenarios are supported:
1. Single file broken into chunks of fixed or variable sizes (chunk size controlled by specific columns)
1. Multiple files. Each file is read into memory as a whole
1. Multiple files. Each file is broken to chunks of fixed or variable sizes
1. Multiple files. Files are grouped to different groups with each group read into memory simultaneously with no file breaking into chunks
1. Multiple files. Same as 4) but files within each group are broken to chunks of fixed or variable sizes

## How to use?
You need to supply your own data analysis routines. Some input and output formats must be followed in order for this program to functional properly, i.e. both the input and output must be a list of tuples with each tuple having three fields: file_path, chunk_id, DataFrame. File_path and chunk_id uniquely defines one chunk from one file, thus they are included in the input. In addition, the output must also be a list of tuples. Both file_path and chunk_id are especially useful for identifying the final summary returned by the process routine. Note, extra inputs are allowed for your process routines as long as it is passed in as a dictionary. As an example, I have included one dummy routine ("count_rows") in the main source file

## What is this module good for and what is it not good for?
This module is especially good for processing large data sets present in a single (physical or virtual) machine utilizing as many cores as possible. There is no upper limit of the size of the data sets as long as the intermediate results do not blow up the memory

## Why not Dask?
First of all, this program is light-weighted and only depends on a few core modules, such as Pandas, Numpy, and Multiprocessing. For those who are familiar with these modules, learning curve is little. On the other hand, Dask still demands some learning efforts. Secondly, each chunk that is passed to the processing routine is a legitimate Pandas DataFrame, hence all Pandas operations are supported by default. However, this is not the case with Dask

## Under the hood
At the run state, this program has one primary process dedicated to task management, i.e., file reading, workload distributing, and summary saving. The main process spawns a few slave processes for data analysis in parallel. The number of slave processes can be user-specified or automatically assigned. The communication between the main process and the slave processes are through two process-safe queues: the data queue transfers the raw data from the main process to the slave processes; the result queue works in the opposite way and transfers the processed results from the slave processes to the main process
