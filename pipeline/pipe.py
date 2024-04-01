from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda
from .worker import Task, create_workers
from .partition import _split_module
import math

# ASSIGNMENT 4.2
def _clock_cycles(num_batches: int, num_partitions: int) -> Iterable[List[Tuple[int, int]]]:
    '''Generate schedules for each clock cycle.

    An example of the generated schedule for m=3 and n=3 is as follows:
    
    k (i,j) (i,j) (i,j)
    - ----- ----- -----
    0 (0,0)
    1 (1,0) (0,1)
    2 (2,0) (1,1) (0,2)
    3       (2,1) (1,2)
    4             (2,2)

    where k is the clock number, i is the index of micro-batch, and j is the index of partition.

    Each schedule is a list of tuples. Each tuple contains the index of micro-batch and the index of partition.
    This function should yield schedules for each clock cycle.
    '''
    # BEGIN SOLUTION
    schedule = [[] for x in range(num_batches + num_partitions-1)]
    for batch_idx in range(num_batches):
        for partition_idx in range(num_partitions):
            schedule[batch_idx + partition_idx].append((batch_idx, partition_idx))
    return schedule
    # END SOLUTION

class Pipe(nn.Module):
    def __init__(
        self,
        module: nn.ModuleList,
        split_size: int = 1,
    ) -> None:
        super().__init__()

        self.split_size = int(split_size)
        self.partitions, self.devices = _split_module(module)
        (self.in_queues, self.out_queues) = create_workers(self.devices)

    # ASSIGNMENT 4.2
    def forward(self, x):
        ''' Forward the input x through the pipeline.

        Hint:
        1. Divide the input mini-batch into micro-batches.
        2. Generate the clock schedule.
        3. Call self.compute to compute the micro-batches in parallel.
        4. Concatenate the micro-batches to form the mini-batch and return it.
        '''
        # BEGIN SOLUTION
        #x = x.unsqueeze(1)
        #self.microbatch_size = 64
        #schedule = _clock_cycles(len(x), len(self.partitions))
        #schedule = _clock_cycles(1, len(self.partitions))
        #import math
        #num_microbatches = int(len(x)/self.microbatch_size)
        #if(len(x) % self.microbatch_size != 0):
        #    num_microbatches += 1
        schedule = _clock_cycles(self.split_size, len(self.partitions))
        x = self.compute(x, schedule)
        return torch.concat(x)
        # END SOLUTION

    # ASSIGNMENT 4.2
    def compute(self, batch, schedule: List[Tuple[int, int]]) -> None:
        '''Compute the micro-batches in parallel.

        Hint:
        1. Retrieve the partition and microbatch from the schedule.
        2. Use Task to send the computation to a worker. 
        3. Use the in_queues and out_queues to send and receive tasks.
        4. Store the result back to the batches.
        '''
        partitions = self.partitions
        devices = self.devices
        microbatch_size = math.ceil(len(batch)/self.split_size)

        # BEGIN SOLUTION
        out_list = []
        output_len = 0
        batch_len = len(batch)
        wait_len = 0
        def execute_forward(partition, inp):
            #print(type(inp))
            def inner_func():
                return partition(inp)
            return inner_func #lambda: partition(inp)
        def extract_queue_result(x):
            if(x[0]):
                return x[1][1]
            else:
                print("ERROR: {}".format(x[1]))
                exit()
                
        for clock_cycle_list in schedule:
            for (batch_idx, partition_idx) in clock_cycle_list:
                if(partition_idx == 0):
                    #x = batch[batch_idx] #batch[batch_idx*microbatch_size : (batch_idx+1)*microbatch_size]
                    #x = batch
                    x = batch[batch_idx*microbatch_size : (batch_idx+1)*microbatch_size]
                    wait_len += len(x)
                else:
                    x = extract_queue_result(self.out_queues[partition_idx-1].get())
                self.in_queues[partition_idx].put(Task(execute_forward(partitions[partition_idx], x.to(devices[partition_idx]))))                    
                        
        #while(len(out_list) < batch_len):
        #while(output_len < batch_len):
        while(output_len < wait_len):
            #print((output_len, batch_len))
            out_list.append(extract_queue_result(self.out_queues[len(partitions)-1].get()))
            output_len += len(out_list[-1])
        if(len(out_list) == 0):
            import pdb
            pdb.set_trace()
        return out_list                                       
        # END SOLUTION

