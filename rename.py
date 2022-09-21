import os

PathName = "C:/Users/aki/Documents/dataset/trainB/"

i=0

for item in os.listdir(PathName):
    os.rename(os.path.join(PathName,item),os.path.join(PathName,("Cartoon"+str(i)+".jpg")))
    i += 1
