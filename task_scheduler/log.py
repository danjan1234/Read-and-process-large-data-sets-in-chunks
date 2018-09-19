"""
This module is designated to log writing

Author: Justin Duan
Time: 2018/08/29 5:35PM
"""

from multiprocessing import Manager
from datetime import datetime
import time


class Log:
    def __init__(self, log_path, log_interval=10):
        """
        Arguments:
            log_interval: the default interval in seconds between two log writings
        """
        self._log_path = log_path
        self._log_interval = log_interval
        self._log_queue = Manager().Queue()
        self._prev_time = time.time()
        self._log_mode = 'a'

        self.put()
        self.put("#### Log starts ####")

    def put(self, message=""):
        """
        Print the message to the screen and add it to the log queue
        """
        dt = datetime.now()
        dt_str = "{}/{}/{} {}:{}:{}".format(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
        message = '<{}> {}'.format(dt_str, message)
        print(message)
        self._log_queue.put(message)
        
    def write(self, log_interval=None):
        """
        Write the log to the disk
        """
        if log_interval is None:
            log_interval = self._log_interval

        # Skip if the previous save is less than 10s ago
        curr_time = time.time()
        if curr_time - self._prev_time < log_interval:
            return
        self._prev_time = curr_time

        # Save the logs
        logs = []
        while self._log_queue.qsize() > 0:
            logs.append(self._log_queue.get() + '\n')
        if len(logs) > 0:
            with open(self._log_path, self._log_mode) as log_f:
                log_f.writelines(logs)
                self._log_mode = 'a'


if __name__ == '__main__':
    # A few test runs
    print("\nTest logging:")
    log_path = r'C:\Users\justin.duan\Documents\Projects\Chip data\Chip_test_scripts - dev\test_data_set\log.txt'
    log = Log(log_path, log_interval=2)
    for i in range(5):
        log.put(str(i))
    log.write(log_interval=0)
