import os
filepath = os.getcwd()
def MakeFile(file_name):
    temp_path = filepath + file_name
    with open(file_name, 'w') as f:
    f.write('''\
def print_success():
    print "sucesss"        
''')
    print 'Execution completed.'

MakeFile("bla.txt");