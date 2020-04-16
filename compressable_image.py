import numpy as np

class compressable_image:
    def __init__(self, image):
        
        self.M = len(image)
        self.N = len(image[0])
        self.unrolled_image = self.unroll_image(image)
        self.image = []
        
    def unroll_image(self, original_image):
        
        new_image = np.asarray(original_image).ravel()
        
        return new_image
    
    def k_means_compression(self, K=16, iteration_limit=10):
        
        # ||x-mu||**2
        euclidean_distance = lambda x: np.inner(x,x)
        
        centroids = np.random.randint(255, size=(K, 3))
        last = np.zeros(self.M*self.N)
        assignments = np.zeros(self.M*self.N)
        changed = True
        iteration = 0
        while iteration < iteration_limit:
            for i, pixel in enumerate(self.unrolled_image):  
                assigned_k = -1
                closest = float('inf')
                for k, centroid in enumerate(centroids):
                    dist = euclidean_distance(pixel-centroids[k])
                    if dist < closest:
                        assigned_k, closest = k, dist
                assignments[i] = assigned_k
        
            if np.array_equal(last, assignments):
                break
            else:
                last = np.copy(assignments)
                assignments = np.zeros(self.M*self.N)
                
            for i, centroid in enumerate(centroids):
                points = [self.unrolled_image[x] for x in range(len(last)) if last[x] == i]
                if len(points) != 0:
                    new_centroid = np.sum(points, axis = 0)/len(points)
                    centroids[i] = new_centroid
            iteration += 1
            print(iteration)
        
        for pixel in range(len(self.unrolled_image)):
            self.unrolled_image[pixel] = centroids[int(last[pixel]),:]

        self.reroll_image()
        
    def reroll_image(self):
        
        self.image = []
        ind = 0
        for i in range(self.M):
            self.image.append([])
            for j in range(self.N):
                self.image[-1].append([])
                for d in range(3):
                    self.image[-1][-1].append(self.unrolled_image[ind][d])
                ind += 1
        self.image = np.asarray(self.image).astype(np.uint8)

    #def normalize(self):
    #    
    #    
    #    
    #def pca(self, var_ret=99):
