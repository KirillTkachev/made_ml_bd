#!/usr/bin/env python3
import sys

chunk = 0
mean = 0
var = 0

for line in sys.stdin:
    current_chunk, current_mean, current_var = line.split(' ')
    current_chunk = int(current_chunk)
    current_var = float(current_var)
    current_mean = float(current_mean)
    var = (chunk * var + current_chunk * current_var) / (chunk + current_chunk) \
        + chunk * current_chunk \
        * ((mean - current_mean) / (chunk + current_chunk)) ** 2
    mean = (chunk * mean + current_chunk * current_mean) / (chunk + current_chunk)
    chunk += current_chunk

print(var)


