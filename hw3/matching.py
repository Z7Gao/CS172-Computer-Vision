import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance
from matplotlib.patches import ConnectionPatch
import numpy as np
from scipy.linalg import solve, lstsq, norm, svd

THRES = 100

class Matching:
	def __init__(self, path1, path2):
		# 1->2
		img1, img2 = cv2.imread(path1, 1), cv2.imread(path2,1)

		self.img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
		self.img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

		self.img1_rgb = cv2.resize(self.img1_rgb, (320,240))
		self.img2_rgb = cv2.resize(self.img2_rgb, (320,240))

		self.img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
		self.img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

		self.img1 = cv2.resize(self.img1, (320,240))
		self.img2 = cv2.resize(self.img2, (320,240))

		self.surf_extraction()
		self.__surf_show()
		self.matching()
		self.__matching_show()

	def surf_extraction(self):
		# show surf's result
		surf = cv2.xfeatures2d.SURF_create()
		self.kp1, self.des1 = surf.detectAndCompute(self.img1, None)
		self.kp2, self.des2 = surf.detectAndCompute(self.img2, None)
		print(len(self.kp1),len(self.kp2))

	def __surf_show(self):
		kp_img1 = cv2.drawKeypoints(self.img1_rgb, self.kp1[0:100], None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		kp_img2 = cv2.drawKeypoints(self.img2_rgb, self.kp2[0:100], None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		fig = plt.figure()
		ax1 = fig.add_subplot(1, 2, 1)
		plt.imshow(kp_img1)
		ax2 = fig.add_subplot(1, 2, 2)
		plt.imshow(kp_img2)
		plt.show()

	def matching(self):
		self.pairs = [] # the index pair of 2 images' keypoints
		for ind1, val1 in enumerate(self.des1):
			min, min_kp = distance.euclidean(self.des2[0], val1), 0
			sec_min, sec_min_kp = distance.euclidean(self.des2[1], val1), 1
			if sec_min < min:
				sec_min, min = min, sec_min
				sec_min_kp, min_kp = min_kp, sec_min_kp
			for ind2, val2 in enumerate(self.des2):
				tmp_dis = distance.euclidean(val2, val1)
				if tmp_dis >= sec_min:
					continue
				elif min <= tmp_dis < sec_min:
					sec_min = tmp_dis
					sec_min_kp = ind2
					# print("sec_min change to {} of index {}".format(sec_min, sec_min_kp) )
				elif tmp_dis < min:
					min = tmp_dis
					min_kp = ind2   
					
			if min < 0.75*sec_min:
				self.pairs.append((ind1, min_kp))

		print(len(self.pairs))    
		# match[j] is one from kp2 that corresponds to kp1_new[j]
		self.match = [(self.kp1[i], self.kp2[j]) for i,j in self.pairs]   
		self.match_cntr = [(tuple(self.kp1[i].pt), tuple(self.kp2[j].pt)) for i,j in self.pairs]   
		self.match_num = len(self.match)

	def __matching_show(self):
		fig = plt.figure()
		ax1 = fig.add_subplot(1, 2, 1)
		plt.imshow(self.img1_rgb)
		ax2 = fig.add_subplot(1, 2, 2)
		plt.imshow(self.img2_rgb)

		for i in self.match_cntr:
			con = ConnectionPatch(xyA=i[0], xyB=i[1], coordsA="data", coordsB="data",
						axesA=ax1, axesB=ax2, color="red")
			ax2.add_artist(con)
			ax1.plot(i[0][0], i[0][1],'bo', markersize=1)
			ax2.plot(i[1][0], i[1][1],'bo', markersize=1)
		plt.show()

	def ransac(self, N):
		final_H, final_inlier, final_avg = 0, 0, 0
		for i in range(N):
			# find 4 pairs randomly
			np.random.seed(i)
			random = np.random.randint(0, self.match_num, 4)
			four_pairs = [self.match_cntr[i] for i in random]
			
			# fit these 4 pairs into a Homography matrix
			H_i, inlier_num = Matching.compute_homography(four_pairs)
			if inlier_num >= 3 :
				# compute error of all pairs based on the homography matrix
				error, inlier_num, inlier_list = self.compute_error(H_i)					
				if inlier_num > final_inlier:
					# print("H_[-1,-1]=",H_[-1,-1])
					final_inlier = inlier_num
					new_pairs = [val for ind, val in enumerate(self.match_cntr) if inlier_list[ind]]
					H_, inlier_num = Matching.compute_homography(new_pairs)
					final_H = H_ 
					final_avg = error/inlier_num
					

		self.H = final_H
		self.inlier_num = final_inlier
		self.avg = final_avg


	def compute_error(self,H,flag=None):
		e = 0
		i = 0
		inlier_list = [0]*self.match_num
		if flag == None:
			for pair in self.match_cntr:
				pi = np.array([pair[0][0],pair[0][1],1])
				pi_ = np.array([pair[1][0],pair[1][1],1])
				tmp = norm(H*pi/H*pi[-1] - pi_)
				# print(tmp,i)
				# int_pair = ((int(pair[0][0]),int(pair[0][1])),(int(pair[1][0]),int(pair[1][1])))
				# print("error of pair {} is {}".format(int_pair, tmp))
				if tmp <= THRES:
					e += tmp
					inlier_list[i] = 1
				i += 1
			return e, sum(inlier_list), inlier_list
		else:
			for pair in flag:
				pi = np.array([pair[0][0],pair[0][1],1])
				pi_ = np.array([pair[1][0],pair[1][1],1])
				tmp = norm(H*pi/H*pi[-1] - pi_)
				# print(tmp,i)
				# int_pair = ((int(pair[0][0]),int(pair[0][1])),(int(pair[1][0]),int(pair[1][1])))
				# print("error of pair {} is {}".format(int_pair, tmp))
				if tmp <= THRES:
					e += tmp
					i += 1
			return e, i



	@staticmethod
	def compute_homography(pairs):
		# compute A, size n*8
		n = 2*len(pairs)
		# A = np.eye(n, dtype=np.int32)
		A = np.zeros((n,9))
		j = 0
		for p in pairs:
			xi, yi = p[0]
			xi_, yi_ = p[1]		
			A[j] = np.array([-xi, -yi, -1, 0, 0, 0, xi_*xi, xi_*yi, xi_])
			A[j+1] = np.array([0,  0,  0, -xi, -yi, -1, yi_*xi, yi_*yi, yi_])	
			j += 2
		_,_,v = svd(A)
		h = np.reshape(v[8],(3,3))
		h = h/h[-1,-1]

		i = 0
		for p in pairs:
			print(norm(np.dot(h,[p[0][0],p[0][1],1])-[p[1][0],p[1][1],1]))
			if norm(np.dot(h,[p[0][0],p[0][1],1])-[p[1][0],p[1][1],1]) <= THRES:
				i += 1
		return h, i


if __name__ == "__main__":
	a = Matching('./lib/lib1.jpg','./lib/lib2.jpg')   
	a.ransac(10000)
	print(a.inlier_num, a.avg, a.H)



'''
		# compute A, size n*9
		n = 2*len(pairs)+1
		# A = np.eye(n, dtype=np.int32)
		A = np.zeros((n,9))
		A[-1,-1] = 1
		j = 0
		for p in pairs:
			xi, yi = p[0]
			xi_, yi_ = p[1]
			A[j] = np.array([xi, yi, 1, 0,  0,  0, -xi_*xi, -xi_*yi, -xi_])
			A[j+1] = np.array([0,  0,  0, xi, yi, 1, -yi_*xi, -yi_*yi, -yi_])			
			j += 2
		# compute b, 9*1. the last line for constraing H.
		b = np.zeros((n,1), dtype=np.int32)
		b[-1] = 1
		# print("A={},b={}".format(A,b))
		# solve H for AH = b
		H = lstsq(A,b)	
		H = H[0]
		# print("Ah = b residual: ",norm(A*H[0]-b))
		H = np.array([ [H[0][0],H[1][0],H[2][0]], [H[3][0],H[4][0],H[5][0]], [H[6][0],H[7][0],H[8][0]] ])
		# print("H=",H)

		try:
			H = H/H[-1,-1]
		except ZeroDivisionError:
			print("ao!s")
			return [],0
		i = 1
		pair_residual = []
		for pair in pairs:
			pi = np.array([pair[0][0],pair[0][1],1])
			pi_ = np.array([pair[1][0],pair[1][1],1])
			tmp = norm(H*pi - pi_)
			pair_residual.append(tmp)
			# print("for pair {}, |p'-hp| = {}".format(i,tmp))
			i += 1
		# print("pair residual max is {}".format(max(pair_residual)))
		pair_residual.sort()
		# inlier = max(pair_residual)
		if len(pair_residual) > 4:
			for i in range(1,len(pair_residual)):
				if 0.9*pair_residual[-i] < pair_residual[-(i+1)]:
					return H, pair_residual[-i]
		else:c
			if 0.95*pair_residual[-1] < pair_residual[-2]:
				return H, pair_residual[-1]
			else:
				return H, pair_residual[-2]
'''


class testing_matchers:
    def __init__(self):
        self.surf = cv2.xfeatures2d.SURF_create()
        self.flann = cv2.FlannBasedMatcher(dict(algorithm = 0, trees = 5), dict(checks = 100))

    def match(self, i1, i2, direction=None):
        im1_gray = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
        im1_kp, im1_des = self.surf.detectAndCompute(im1_gray, None)

        im2_gray = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
        im2_kp, im2_des = self.surf.detectAndCompute(im2_gray, None)

        matches = self.flann.knnMatch(im2_des, im1_des, k=2)
        good = []
        for _, (m, n) in enumerate(matches):
            if m.distance < 0.75*n.distance:
                good.append((m.trainIdx, m.queryIdx))

        if len(good) >= 4:
            matched_im2_kp = np.float32([im2_kp[i[1]].pt for i in good])
            matched_im1_kp = np.float32([im1_kp[i[0]].pt for i in good])
            H, _ = cv2.findHomography(matched_im2_kp, matched_im1_kp, cv2.RANSAC, 4)
            return H
        return None
