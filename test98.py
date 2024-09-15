
Str = input("please input a string:\n").upper()

fp = open("test.txt", "w+")
fp.write(Str)
fp = open("test.txt", "w+")
print(fp.read())
fp.close()