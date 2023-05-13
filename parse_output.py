import sys
import os

task = sys.argv[1]
model = sys.argv[2] + '-'

files = os.listdir(task)

files = [f for f in files if f.startswith(model)]

d = []
mean = []
std = []
constant = []

for file in files:
    with open(os.path.join(task, file), 'r') as f:
        lines = f.readlines()
        lines = [ line.strip() for line in lines  ]
    start_index = file.index('-') + 1
    end_index = file.index('.')
    d.append( file[ start_index:end_index  ]  )
    for line in lines:
        if 'Constant classifier: ' in line:
            constant.append( line[ len( 'Constant classifier: ' ) :  ]  )
        if 'Final test results: ' in line:
            line = line.split(' ')
            mean.append( line[1]  )
            std.append( line[2]  )
        d.append(   )
print( mean  )
print( std  )
print( constant  )
