from matching import Matching, testing_matchers
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

import cv2
import numpy as np 

class Stitching:
    def __init__(self):
        self.im1 = self.__load_im('./lib/lib1.jpg')
        self.im2 = self.__load_im('./lib/lib2.jpg')
        self.im3 = self.__load_im('./lib/lib3.jpg')
        self.im4 = self.__load_im('./lib/lib4.jpg')

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_xlabel("I1")
        plt.imshow(self.im1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_xlabel("I2")

        plt.imshow(self.im3)
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_xlabel("I3")

        plt.imshow(self.im4)
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_xlabel("I4")

        plt.imshow(self.im2)
        plt.show()

        self.matcher = testing_matchers()

        self.im12, off1 = self.left_stitch(self.im1, self.im2, 1)
        self.im123, off2 = self.left_stitch(self.im12, self.im3)
        self.im1234 = self.right_stitch(self.im123, self.im4, (off1[0]+off2[0], off1[1]+off2[1]) )

    def left_stitch(self, a, b, flag = None):
        if flag == 1:
            m = Matching('./lib/lib1.jpg','./lib/lib2.jpg')
            m.ransac(100)
            H_inv = m.H

        H = self.matcher.match(a, b, 'left')
        H_inv = np.linalg.inv(H)
        
        # find top left point for offset calculation.
        tl = np.dot(H_inv, np.array([0,0,1]))
        tl = tl/tl[-1]
        H_inv[0][-1] += abs(tl[0])
        H_inv[1][-1] += abs(tl[1])

        # find down right for size calculation.
        w, h = a.shape[1], a.shape[0]
        dr = np.dot(H_inv, np.array([w, h, 1]))
        dsize = (int(dr[0])+abs(int(tl[0])), int(dr[1]) + abs(int(tl[1])))

        # warp a into b's view and put them together.
        merge = cv2.warpPerspective(a, H_inv, dsize)
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        plt.imshow(merge)
        ax2 = fig.add_subplot(1, 2, 2)
        merge[abs(int(tl[1])):b.shape[0]+abs(int(tl[1])), abs(int(tl[0])):b.shape[1]+abs(int(tl[0])), :] = b
        plt.imshow(merge)
        plt.show()
        offset = (abs(int(tl[0])),abs(int(tl[1])))
        return merge, offset

    def right_stitch(self, a, b, off):
        H = self.matcher.match(a, b, 'right')

        # find down right for size calculation.
        dr = np.dot(H, np.array([b.shape[1], b.shape[0], 1]))
        dr = dr/dr[-1]
        dsize = (int(dr[0])+a.shape[1], int(dr[1])+a.shape[0])

        # warp a into b's view and put them together.
        merge = cv2.warpPerspective(b, H, dsize)
        merge[0:a.shape[0],0:off[0]+500,:] = a[0:a.shape[0],0:off[0]+500,:]
        print(a.shape)

        plt.imshow(merge)
        plt.show()

        return merge


    @staticmethod
    def __load_im(path):
        img = cv2.imread(path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1280,960))
        return img

        
if __name__ == "__main__":
    s = Stitching()

'''
    def left_stitch_2(self, im_l, im_r):
        w, h = self.im1.shape[1], self.im1.shape[0]
        print("w={},h={}".format(w,h))
        # find transposed corners and normalize them 
        # print(self.H21,self.H21.shape)
        corners = np.dot(self.H21,np.array([[0,0,w,w],
                                            [0,h,0,h],
                                            [1,1,1,1]]))
        tl, dl, tr, dr = [i/i[-1] for i in list(corners.T)]
        print("tl={},\n dl={},\n tr={},\n dr={}".format(tl,dl,tr,dr))
        dsize_x = ceil(abs(max(tr[0],dr[0]) - min(tl[0],dl[0]))) #?
        dsize_y = ceil(abs(max(dl[1],dr[1]) - min(tl[1],tr[1]))) #?
        dsize = (dsize_x, dsize_y)
        print(dsize)
        # M = np.float32([[1, 0, ceil(-tl[0])], [0, 1, ceil(-tl[1])], [0,0,1]])
        # visualize warped img1 
        # M = np.dot(M,self.H21)
        M = self.H21.copy()
        #M[0,-1] += ceil(-tl[0])
        #M[1,-1] += ceil(-tl[1])
        self.im1_warped = cv2.warpPerspective(self.im1, M, (dsize_x,dsize_y))
        # self.im1_warped_trans = cv2.warpAffine(self.im1_warped, M, dsize)
        # print(self.im1_warped.shape)
        fig = plt.figure()
        ax1 = fig.add_subplot(1,3,1)
        plt.imshow(self.im1_warped)
        ax2 = fig.add_subplot(1,3,2)
        plt.imshow(self.im2)    
        ax2.set_xticks([0,200,400,613])
        ax2.set_yticks([0,250,500,732])

        ax3 = fig.add_subplot(1,3,3)
        M = self.H21.copy()
        M = np.dot(np.float32([[1, 0, ceil(-tl[0])], [0, 1, ceil(-tl[1])], [0,0,1]]),M)
        self.im1_warped = cv2.warpPerspective(self.im1, M, (dsize_x,dsize_y))

        plt.imshow(self.im1_warped) 
        plt.show()
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1)
        plt.imshow(self.im1_warped)
        ax2 = fig.add_subplot(1, 3, 2)
        M = np.float32([[1, 0, ceil(-tl[0])], [0, 1, ceil(-tl[1])]])
        self.im2_trans = cv2.warpAffine(self.im2, M, (dsize_x+self.im2.shape[0],dsize_y+self.im2.shape[1]))
        plt.imshow(self.im2_trans)
        ax2 = fig.add_subplot(1, 3, 3)
        self.im12 = np.zeros(self.im2_trans.shape,dtype=np.uint8)
        
        self.im12[0:dsize_y,0:dsize_x,:] = self.im1_warped.astype(np.uint8)
        self.im12[ceil(-tl[1]):ceil(-tl[1])+self.im2.shape[1],ceil(-tl[0]):ceil(-tl[0])+self.im2.shape[0],:] \
            = self.im2_trans[ceil(-tl[1]):ceil(-tl[1])+self.im2.shape[1],ceil(-tl[0]):ceil(-tl[0])+self.im2.shape[0],:].astype(np.uint8)
        plt.imshow(self.im12)
        plt.show()
        """

    '''
