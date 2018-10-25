import os

def main():
    class_names_fid = open('class_list.txt')
    for class_name in class_names_fid.readlines():
        print './gt/' + class_name.split('\n')[0] + '_test_gt.txt'

if __name__ == '__main__':
    main()
