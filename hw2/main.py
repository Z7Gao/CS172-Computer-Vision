import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report


from sklearn import svm
import scipy.cluster.vq as vq
import copy

# classfication method option
LINEAR = 1
HISTORGRAM = 2

# step size of computing sift 
STEP_SIZE = 16

class Image_Classification:
    def __init__(self,train_path, test_path):

        print("loading dataset from {} ...".format(train_path))
        self.train_set, self.train_labels = Image_Classification.__load_data(train_path)
        
        print("loading dataset from {} ...".format(test_path))
        self.test_set, self.test_labels = Image_Classification.__load_data(test_path)
        
        print("loading labels ...".format(train_path))
        self.labels = Image_Classification.__load_label()


    def feature_extraction(self, step):

        self.step = step
        print("extracting descriptors from training set...")
        _, self.train_descriptors = zip(*[self.__feature_extraction_per_image(img) for img in self.train_set])
        del _
        import gc
        gc.collect()
        self.train_descriptors =  np.asarray(self.train_descriptors)
        print("self.train_descriptors.sshape = {}".format(self.train_descriptors.shape))
        print("extracting descriptors from testing set...")
        _, self.test_descriptors = zip(*[self.__feature_extraction_per_image(img) for img in self.test_set])
        del _
        import gc
        gc.collect()
        self.test_descriptors = np.asarray(self.test_descriptors)
        print("self.test_descriptors.shape = {}".format(self.test_descriptors.shape)) 
        
        self.train_features = np.vstack((copy.deepcopy(i) for i in self.train_descriptors))
        self.test_features = np.vstack((copy.deepcopy(i) for i in self.test_descriptors))
        print("restacked self.train_features.shape = {}".format(self.train_features.shape))


    def __feature_extraction_per_image(self, img):
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints, descriptors = sift.compute(img, [cv2.KeyPoint(x, y, self.step) for y in range(0, img.shape[0], self.step) for x in range(0, img.shape[1], self.step)])
        return keypoints, descriptors


    def codebook_generation(self, k):

        self.k = k
        print("generating codebook with k = {}...".format(k))
        self.kmeans = MiniBatchKMeans(n_clusters=k).fit(self.train_features)
        
        print("extracting visual words from means...")
        self.codebook = self.kmeans.cluster_centers_.squeeze()   
        
        print("codebook size = {}".format(self.codebook.shape))


    def bow_feature_quantization(self):

        print("quantizing training images in training set...")
        self.train_hist = np.array([np.bincount(self.kmeans.predict(copy.deepcopy(i)), minlength = self.k) for i in self.train_descriptors])
       
        print("quantizing training images in testing set...")
        self.test_hist = np.array([np.bincount(self.kmeans.predict(copy.deepcopy(i)), minlength = self.k) for i in self.test_descriptors])
        print("histogram shape:", self.train_hist.shape, self.test_hist.shape)
        
        print("normalizing the encoding image data...")
        scalar = StandardScaler().fit(self.train_hist)
        self.train_hist = scalar.transform(self.train_hist)
        self.test_hist = scalar.transform(self.test_hist)

    def bow_classfication(self, method):

        # method is in {LINEAR, HISTOGRAM}
        if method == LINEAR:
            C = np.logspace(-3, 3, 10)
            print("adjusting hyper-parameters...")
            clf = GridSearchCV(svm.LinearSVC(), dict(C=C), cv=5, n_jobs=-1, refit = True)
            clf.fit(self.train_hist, self.train_labels)
            
            print("YES! Best parameters set found:")
            print(clf.best_estimator_)
            
            print("traing the svm...")
            clf.fit(self.train_hist, self.train_labels)
            
            print("testing on the testing set...")
            self.test_predict = clf.predict(self.test_hist)
            
            print("generating testing result...")
            print(classification_report(self.test_labels, self.test_predict, target_names=Image_Classification.__load_label()))



    def spm_feature_quantization(self, step, level):

        self.step = step

        # level: 1 or 2
        self.level = level
        self.train_descriptors_spm = []
        self.test_descriptors_spm = []
        self.train_hist_spm = []
        self.test_hist_spm = []
        
        print("training set sift extraction...")
        x = y = 0
        
        for img in self.train_set:
            h = []
            for l in range(level+1):
                # print("level = {}".format(l))
                step = int(256/(2**l))
                y = 0
                for i in range(1, 2**l+1):
                    x = 0
                    for j in range(1, 2**l+1):
                        desc = self.__feature_extraction_per_image(copy.deepcopy(img[y:y+step, x:x+step]))[1]                
                        predict = self.kmeans.predict(desc)
                        histo = np.bincount(predict, minlength=self.k).reshape(1,-1).ravel()
                        weight = 2**(l-level)
                        h.append(weight*histo)
                        x += step
                    y += step
                
            hist = np.array(h).ravel()
            hist = (hist-np.mean(hist))/np.std(hist)
            self.train_hist_spm.append(hist)


        print("testing set sift extraction...")
        x = y = 0
        
        for img in self.test_set:
            h = []
            for l in range(level+1):
                # print("level = {}".format(l))
                step = int(256/(2**l))
                y = 0
                for i in range(1, 2**l+1):
                    x = 0
                    for j in range(1, 2**l+1):
                        desc = self.__feature_extraction_per_image(copy.deepcopy(img[y:y+step, x:x+step]))[1]                
                        predict = self.kmeans.predict(desc)
                        histo = np.bincount(predict, minlength=self.k).reshape(1,-1).ravel()
                        weight = 2**(l-level)
                        h.append(weight*histo)
                        x += step
                    y += step
                
            hist = np.array(h).ravel()
            hist = (hist-np.mean(hist))/np.std(hist)
            self.test_hist_spm.append(hist)

        print(np.array(self.train_hist_spm).shape)
        print(np.array(self.test_hist_spm).shape)


    def spm_classification(self, method = LINEAR):
        C = np.array([0.001, 0.01, 0.1, 1, 10, 100, 1000])
        print("adjusting hyper-parameters...")
        clf = GridSearchCV(svm.LinearSVC(), dict(C=C), cv=5, n_jobs=-1, refit = True)
        clf.fit(self.train_hist_spm, self.train_labels)
        
        print("YES! Best parameters set found:")
        print(clf.best_estimator_)
        
        print("traing the svm...")
        clf.fit(self.train_hist_spm, self.train_labels)
        
        print("testing on the testing set...")
        self.test_predict = clf.predict(self.test_hist_spm)
        
        print("generating testing result...")
        print(classification_report(self.test_labels, self.test_predict, target_names=Image_Classification.__load_label()))




    @staticmethod
    def __load_data(path):
        with open(path, 'r') as f:
            dataset = f.readlines()
        imgs, labels = [], []
        for data in dataset:
            path, label = data.strip().split(' ')
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # print("loading image from {}".format(path))
            if img.shape[:2] != (128,128):
                img = cv2.resize(img, (128,128))
            imgs.append(img)
            labels.append(label)
        return imgs, labels

    @staticmethod
    def __load_label():
        with open("./label.txt",'r') as f:
            names = f.readlines()
        return [n.strip() for n in names]   


   

if __name__ =="__main__":
    print("==============================Let's start!======================================")
    
    hdlr = Image_Classification('./train.txt','./test.txt')
    hdlr.feature_extraction(8)
    hdlr.codebook_generation(2000)
    hdlr.bow_feature_quantization()
    hdlr.bow_classfication(LINEAR)
    hdlr.spm_feature_quantization(8,1)
    hdlr.spm_classification()
    hdlr.spm_feature_quantization(8,2)
    hdlr.spm_classification()

