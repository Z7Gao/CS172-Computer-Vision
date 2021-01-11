import cv2
import numpy as np
from scipy.sparse import dia_matrix, block_diag, lil_matrix
from scipy.sparse.linalg import cg

ORIGIN,AVERAGE,MIXED = 0,1,2

class Poisson:
    def __init__(self,src,msk,trgt,type):
        '''
        src:  a*b RGB,  numpy array
        msk:  a*b Grey, numpy array, normalized
        trgt: m*n RGB,  numpy array
        '''
        self.type = type
        self.msk = msk

        self.src_r = src[:,:,0]
        self.src_g = src[:,:,1]
        self.src_b = src[:,:,2]
        
        self.trgt_r = trgt[:,:,0]
        self.trgt_g = trgt[:,:,1]
        self.trgt_b = trgt[:,:,2]

        self.g_pixels_num = msk.sum()
        # print(self.g_pixels_num)
        # NOT effcient. Need to study how to do functional programming in NumPy:(
        self.g_pixels = []
        for ind, pixel_list in enumerate(msk.tolist()):
            for ind1, pixel in enumerate(pixel_list):
                if pixel:
                    self.g_pixels.append((ind,ind1))

  
    # some helper functions：
           
    def neighbors(self,pixel):

        # Return the indices of neighbors of pixel.
        #
        # It doesn't consider the edge situation, which can always be avoided by 
        # adjusting the size and location of the source or target image!

        x, y = pixel
        return [(x,y-1),(x+1,y),(x,y+1),(x-1,y)]

        


    def is_mask(self,pixel):
        
        # Return a bool value indicating whether pixel is in the white region of mask.
        #
        # This is based on the normalized property of self.msk, which are done in main body.
        
        return self.msk[pixel]

    def is_edge(self,pixel):
    
        # Return a bool value indicating whether pixel is edge.
        #
        # A is edge if A's in the mask while at least one of its neighbors is not.

        return self.is_mask(pixel) and [not self.is_mask(i) for i in self.neighbors(pixel)].count(True)

    def laplacian(self,pixel,layer): 
    
        # Return the divergence of a pixel in specific layer using Laplacian kernel.
        # 
        # Filter + boundary test can make it more concrete, but not implemented because of the same reason as self.is_mask().
    
        return sum([(-1)*layer[i] for i in self.neighbors(pixel)] ) + 4*layer[pixel]


    # Core algorithms: getting A and b for Ax = b

    def laplacian_mtrx(self):
        
        # Compute A in Ax = b.
        
        self.A = lil_matrix((self.g_pixels_num,self.g_pixels_num))
        for i, pixel in enumerate(self.g_pixels):
            self.A[i,i] = 4
            for j in self.neighbors(pixel):
                if j in self.g_pixels:
                    j_ind = self.g_pixels.index(j)
                    self.A[i,j_ind] = -1


    def div_vec(self):

        # Compute b in Ax = b.

        self.b_r = np.zeros(self.g_pixels_num)
        for i, pixel in enumerate(self.g_pixels):
            self.b_r[i] = self.laplacian(pixel,self.src_r)
            if self.type == MIXED:
                '''
                trgt_grdt =  self.laplacian(pixel,self.trgt_r)
                if abs(self.laplacian(pixel,self.trgt_r)) > abs(self.laplacian(pixel,self.src_r)):
                    self.b_r[i] = trgt_grdt
                '''
                self.b_r[i] = (self.laplacian(pixel,self.trgt_r) + self.b_r[i])
            if self.type == AVERAGE:
                self.b_r[i] = (self.laplacian(pixel,self.trgt_r) + self.b_r[i])/2
            if self.is_edge(pixel):
                for j in self.neighbors(pixel):
                    if not self.is_mask(j):
                        self.b_r[i] += self.trgt_r[j]
                

        self.b_g = np.zeros(self.g_pixels_num)
        for i, pixel in enumerate(self.g_pixels):
            self.b_g[i] = self.laplacian(pixel,self.src_g)
            if self.type == MIXED:
                '''
                trgt_grdt =  self.laplacian(pixel,self.trgt_g)
                if abs(self.laplacian(pixel,self.trgt_g)) > abs(self.laplacian(pixel,self.src_g)):
                    self.b_g[i] = trgt_grdt
                '''
                self.b_g[i] = (self.laplacian(pixel,self.trgt_g) + self.b_g[i])
            if self.type == AVERAGE:
                self.b_g[i] = (self.laplacian(pixel,self.trgt_g) + self.b_g[i])/2
            if self.is_edge(pixel):
                for j in self.neighbors(pixel):
                    if not self.is_mask(j):
                        self.b_g[i] += self.trgt_g[j]


        self.b_b = np.zeros(self.g_pixels_num)
        for i, pixel in enumerate(self.g_pixels):
            self.b_b[i] = self.laplacian(pixel,self.src_b)
            if self.type == MIXED:
                '''
                trgt_grdt =  self.laplacian(pixel,self.trgt_b)
                if abs(self.laplacian(pixel,self.trgt_b)) > abs(self.laplacian(pixel,self.src_b)):
                    self.b_b[i] = trgt_grdt
                '''
                self.b_b[i] = (self.laplacian(pixel,self.trgt_b) + self.b_b[i])
            if self.type == AVERAGE:
                self.b_b[i] = (self.laplacian(pixel,self.trgt_b) + self.b_b[i])/2
            if self.is_edge(pixel):
                for j in self.neighbors(pixel):
                    if not self.is_mask(j):
                        self.b_b[i] += self.trgt_b[j]

        # print(self.b_r)

    def blend(self):
        
        # Solve x in Ax = b and formulate the final result.
        
        self.div_vec()
        self.laplacian_mtrx()
        # print(self.A.shape,self.b_r.shape)

        self.x_r = cg(self.A, self.b_r)
        self.x_r = np.rint(self.x_r[0])
        self.result_r = self.trgt_r
        for i,index in enumerate(self.g_pixels):
            self.result_r[index] = max(0,min(255,self.x_r[i]))

        # print(self.x_r.tolist())
        self.x_g = cg(self.A, self.b_g)
        self.x_g = np.rint(self.x_g[0])
        self.result_g = self.trgt_g
        for i,index in enumerate(self.g_pixels):
            self.result_g[index] = max(0,min(255,self.x_g[i]))

        self.x_b = cg(self.A, self.b_b)
        self.x_b = np.rint(self.x_b[0])
        self.result_b = self.trgt_b
        for i,index in enumerate(self.g_pixels):
            self.result_b[index] = max(0,min(255,self.x_b[i]))

        self.result = cv2.merge([self.result_r,self.result_g,self.result_b])        
        



if __name__ == "__main__":
    # simply read 3 images
    src = cv2.imread("figure/src.jpg")
    msk = cv2.imread("figure/msk.jpg",cv2.IMREAD_GRAYSCALE)
    trgt = cv2.imread("figure/trgt.jpg")

    # normalize mask to 0/1 using ufunc sign() 
    msk = np.sign(msk)
    img = Poisson(src,msk,trgt,2)
    img.blend()
    cv2.imwrite('figure/res_mix.png',img.result)

    img = Poisson(src,msk,trgt,1)
    img.blend()
    cv2.imwrite('figure/res_avrg.png',img.result)

    img = Poisson(src,msk,trgt,0)
    img.blend()
    cv2.imwrite('figure/res.png',img.result)

    # simply read 3 images
    src = cv2.imread("figure/src1.jpg")
    msk = cv2.imread("figure/msk1.jpg",cv2.IMREAD_GRAYSCALE)
    trgt = cv2.imread("figure/trgt1.jpg")

    # normalize mask to 0/1 using ufunc sign() 
    msk = np.sign(msk)
    img = Poisson(src,msk,trgt,2)
    img.blend()
    cv2.imwrite('figure/res1_mix.png',img.result)

    img = Poisson(src,msk,trgt,1)
    img.blend()
    cv2.imwrite('figure/res1_avrg.png',img.result)

    img = Poisson(src,msk,trgt,0)
    img.blend()
    cv2.imwrite('figure/res1.png',img.result)


'''
def laplacian_mtrx0(self):
        
            compute A in Ax = b.
        
        # form of A: https://en.wikipedia.org/wiki/Discrete_Poisson_equation
        
        # inefficient!
        diag0 = np.zeros(self.m) + 4
        diag1 = np.zeros(self.m) - 1
        self.D = dia_matrix([diag0, diag1, diag1],[0, -1, 1]).tolil()
        
        # D: m*m
        D = lil_matrix((self.m, self.m))
        D.setdiag(-1, -1)
        D.setdiag(4)
        D.setdiag(-1, 1)
        # A: mn * mn
        self.A = block_diag([D] * self.n).tolil()
        self.A.setdiag(-1, 1*self.m)
        self.A.setdiag(-1, -1*self.m)
        # print(self.A.shape,self.trgt.shape)
'''