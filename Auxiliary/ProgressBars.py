###########################################################################
###########################################################################
## CLI progress basrs used by jinx.                                      ##
##                                                                       ##
## Copyright (C)  2022  Oliver Michael Kamperis                          ##
## Email: o.m.kamperis@gmail.com                                         ##
##                                                                       ##
## This program is free software: you can redistribute it and/or modify  ##
## it under the terms of the GNU General Public License as published by  ##
## the Free Software Foundation, either version 3 of the License, or     ##
## any later version.                                                    ##
##                                                                       ##
## This program is distributed in the hope that it will be useful,       ##
## but WITHOUT ANY WARRANTY; without even the implied warranty of        ##
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          ##
## GNU General Public License for more details.                          ##
##                                                                       ##
## You should have received a copy of the GNU General Public License     ##
## along with this program. If not, see <https://www.gnu.org/licenses/>. ##
###########################################################################
###########################################################################

import os
import threading
from time import sleep
from typing import Optional
import psutil
from tqdm import tqdm

class ResourceProgressBar:
    """
    Simple progress bar which displays memory and CPU usage in postfix.
    """
    
    __slots__ = ("__process",
                 "__postfix",
                 "__progress_bar",
                 "__running",
                 "__cpu_thread")
    
    def __init__(self,
                 initial: int = 0,
                 total: Optional[int] = None,
                 desc: Optional[str] = None,
                 unit: str = "it",
                 leave: bool = False,
                 ncols: int = 180,
                 miniters: int = 1,
                 colour: str = "cyan"
                 ) -> None:
        """
        Create a resouce usage progress bar.
        See `tqdm.tqdm` for a description of parameters.
        """
        
        ## Process variable used for updating resource usage statistics.
        self.__process = psutil.Process(os.getpid())
        
        ## The progress bar itself.
        self.__postfix = {"Mem(Mb)" : self.__get_mem(),
                          "CPU(%)" : self.__get_cpu()}
        self.__progress_bar = tqdm(postfix=self.__postfix,
                                   initial=initial,
                                   total=total,
                                   desc=desc,
                                   unit=unit,
                                   leave=leave,
                                   ncols=ncols,
                                   miniters=miniters,
                                   colour=colour)
        
        ## Variables for running an additional thread which
        ## updates resource statistics ten times per second.
        self.__running: bool = True
        self.__cpu_thread = threading.Thread(target=self.__update)
        self.__cpu_thread.daemon = True
        self.__cpu_thread.start()
    
    def __get_mem(self) -> str:
        "Get current memory usage in megabits"
        return str(int(self.__process.memory_info().rss / (1024 ** 2))).zfill(5)
    
    def __get_cpu(self) -> str:
        "Get cpu usage in percent."
        return format(self.__process.cpu_percent(), "0.2f").zfill(6)
    
    def __update(self) -> None:
        "Target for the update thread."
        while self.__running:
            self.__postfix["Mem(Mb)"] = self.__get_mem()
            self.__postfix["CPU(%)"] = self.__get_cpu()
            sleep(0.1)
    
    def update(self,
               n: int = 1,
               data: Optional[dict[str, str]] = None
               ) -> None:
        """
        Update the progress bar.
        
        Parameters
        ----------
        `n: int = 1` - The number of increments ran since the last update.
        
        `data: dict[str, str] = {}` - An optional dictionary of additional
        statistics to display in the progress bar's postfix, given as a
        mapping between name to value pairs.
        """
        self.__progress_bar.set_postfix(data | self.__postfix
                                        if data is not None
                                        else self.__postfix)
        self.__progress_bar.update(n)
    
    def close(self) -> None:
        "Cleanup and close the progress bar."
        self.__progress_bar.close()
        self.__running = False