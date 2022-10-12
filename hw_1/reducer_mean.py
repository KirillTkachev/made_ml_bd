#!/usr/bin/env python3 

import sys

chunk = 0
mean = 0

for line in sys.stdin:
    current_chunk, current_mean = line.split(' ')
    current_chunk = int(current_chunk)
    current_mean = float(current_mean)
    mean = (chunk * mean + current_chunk * current_mean) / (chunk + current_chunk) 
    chunk += current_chunk
    
print(mean)
