# !/usr/bin/python
# -*- coding:utf-8 -*-
import subprocess
import time

def main():
    p = subprocess.Popen(["python", "-u", "pmctree.py"], stdout= subprocess.PIPE, stderr = subprocess.STDOUT)
    #p = subprocess.Popen(["ls","-l"],stdout = subprocess.PIPE)
    output, err = p.communicate()
    print output
    while p.poll() == None:  # 检查子进程是否已经结束
        print(p.stdout.readline())
        time.sleep(1)
    print(p.stdout.read())
    print('returen code:', p.returncode)

if __name__ == '__main__':
    main()