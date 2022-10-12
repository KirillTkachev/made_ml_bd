#!/usr/bin/env python3 

import sys

counter = 0
sum_value = 0
for line in sys.stdin:
    line = line.split(',')
    try:
        value = int(line[-7])
    except Exception as e:
        continue
    counter += 1
    sum_value += value

mean = sum_value / counter
print(counter, mean)

