#!/usr/bin/env python3

import sys

counter = 0
sum_value = 0
sum_value_squared = 0

for line in sys.stdin:
    line = line.split(',')
    try:
        value = int(line[-7])
    except Exception as e:
        continue
    counter += 1
    sum_value += value
    sum_value_squared += value ** 2
        
mean = sum_value / counter
var = sum_value_squared / counter - mean ** 2

print(counter, mean, var)
