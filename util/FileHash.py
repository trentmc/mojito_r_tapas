"""FileHash.py
"""
import os
import sys
import md5

def hashfile(filename):
	f = file(filename,'rb');
	m = md5.new();
	readBytes = 1024; # read 1024 bytes per time
	while (readBytes):
		readString = f.read(readBytes);
		m.update(readString);
		readBytes = len(readString);
	f.close();
	return m.hexdigest();

