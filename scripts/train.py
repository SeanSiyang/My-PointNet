import os
import sys


ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)




if __name__ == '__main__':
    print(ROOT_DIR)