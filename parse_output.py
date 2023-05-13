import sys
import os

task = sys.argv[1]
model = sys.argv[2] + '-'

files = os.listdir(task)

files = sorted([f for f in files if f.startswith(model)])


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
    d.append( int( file[ start_index:end_index  ]  ))
    c = None
    for line in lines:
        if 'Constant classifier: ' in line:
            c = float(line[ len( 'Constant classifier: ' ) :  ]  )
        if 'Final test results: ' in line:
            line = line.split(' ')
            mean.append( float(line[3]  ))
            std.append( float(line[4]  ))
    constant.append(c)
print( d  )
print( mean  )
print( std  )
print( constant  )
print( len(d), len(mean), len(constant)  )
