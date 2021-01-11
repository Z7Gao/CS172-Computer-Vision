import os

TRAIN_SIZE = 15
CLASS_NUM = 4

datasetpath = "256_ObjectCategories"
dirs = os.listdir(datasetpath)
dirs.sort()
with open(r'label.txt','w',encoding='utf-8') as f:
    for i in dirs[:CLASS_NUM]:
        f.write(i)
        f.write('\n')

num = 0
Matrix = [[] for x in range(257)]
for d in dirs:
    for _, _, filename in os.walk(os.path.join(datasetpath,d)):
        for i in filename:
            Matrix[num].append(os.path.join(os.path.join(datasetpath,d),i))
    num += 1


with open(r'train.txt','w',encoding='utf-8') as f:
    for i in range(CLASS_NUM):
        for j in range(TRAIN_SIZE):
            f.write('./{} {}'.format(Matrix[i][j],str(i))) 
            f.write('\n')


with open(r'test.txt'.format(i),'w',encoding='utf-8') as f:
    for i in range(CLASS_NUM):
        for j in range(TRAIN_SIZE,len(Matrix[i])):
            f.write('./{} {}'.format(Matrix[i][j],str(i)))
            f.write('\n')
