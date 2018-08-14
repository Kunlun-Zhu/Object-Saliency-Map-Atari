from scipy import ndimage
import matplotlib.pyplot as plt
from filter import *
from segment_graph import *
import time
import gym
import cv2


plt.switch_backend('agg')
# add by kunlun I add this so that plt can be used in server(don't need UI)

# --------------------------------------------------------------------------------
# Segment an image:
# Returns a color image representing the segmentation.
#
# Inputs:
#           in_image: image to segment.
#           sigma: to smooth the image.
#           k: constant for threshold function.
#           min_size: minimum component size (enforced by post-processing stage).
#
# Returns:
#           num_ccs: number of connected components in the segmentation.
# --------------------------------------------------------------------------------
def segment(in_image, sigma, k, min_size,saveindex):
    start_time = time.time()
    height, width, band = in_image.shape
    print("Height:  " + str(height))
    print("Width:   " + str(width))
    smooth_red_band = smooth(in_image[:, :, 0], sigma)
    smooth_green_band = smooth(in_image[:, :, 1], sigma)
    smooth_blue_band = smooth(in_image[:, :, 2], sigma)

    # build graph
    edges_size = width * height * 4
    edges = np.zeros(shape=(edges_size, 3), dtype=object)
    num = 0
    for y in range(height):
        for x in range(width):
            if x < width - 1:
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int(y * width + (x + 1))
                edges[num, 2] = diff(smooth_red_band, smooth_green_band, smooth_blue_band, x, y, x + 1, y)
                num += 1
            if y < height - 1:
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int((y + 1) * width + x)
                edges[num, 2] = diff(smooth_red_band, smooth_green_band, smooth_blue_band, x, y, x, y + 1)
                num += 1

            if (x < width - 1) and (y < height - 2):
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int((y + 1) * width + (x + 1))
                edges[num, 2] = diff(smooth_red_band, smooth_green_band, smooth_blue_band, x, y, x + 1, y + 1)
                num += 1

            if (x < width - 1) and (y > 0):
                edges[num, 0] = int(y * width + x)
                edges[num, 1] = int((y - 1) * width + (x + 1))
                edges[num, 2] = diff(smooth_red_band, smooth_green_band, smooth_blue_band, x, y, x + 1, y - 1)
                num += 1
    # Segment
    u = segment_graph(width * height, num, edges, k)

    # post process small components
    for i in range(num):
        a = u.find(edges[i, 0])
        b = u.find(edges[i, 1])
        if (a != b) and ((u.size(a) < min_size) or (u.size(b) < min_size)):
            u.join(a, b)

    num_cc = u.num_sets()
    print(num_cc)
    output = np.zeros(shape=(height, width, 3),dtype='uint8')

    # pick random colors for each component
    colors = np.zeros(shape=(height * width, 3))
    for i in range(height * width):
        colors[i, :] = random_rgb()

    
    for y in range(height):
        for x in range(width):
            comp = u.find(y * width + x)
            print(comp)
            output[y, x, :] = colors[comp, :]
            #print(output[y,x,:])

    elapsed_time = time.time() - start_time
    print(
        "Execution time: " + str(int(elapsed_time / 60)) + " minute(s) and " + str(
            int(elapsed_time % 60)) + " seconds")

    # displaying the result
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(in_image)
    a.set_title('Original Image')
    a = fig.add_subplot(1, 2, 2)
    plt.imshow(output)
    a.set_title('Segmented Image')
    #plt.savefig("./result/space"+str(saveindex)+".png")
    plt.savefig("./result/new"+str(saveindex)+".png")
    #plt.show()
    return u,width,height


if __name__ == "__main__":
    
    #Use for MsPacman
    sigma = 0.3
    k = 300
    min_val = 5
    '''
    sigma = 0.2
    k = 300
    min_val = 3
    '''
    env = gym.make("Frostbite-v0")
    observation = env.reset()
    for i in range(500):
        env.render()
        action = env.action_space.sample() # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        if(i>10):
            print("Loading is done.")
            print("processing...")
            #ob2 = cv2.cvtColor(observation,cv2.COLOR_RGB2BGR)
            plt.imsave('./result/observation'+str(i)+'.png',observation)
            segment(observation, sigma, k, min_val,i)
        
            #Use for Space Invaders
            
            
            
            '''
            sigma=0.35 # try decreasing sigma value
            k=350
            min_val =20
            '''
            #input_path = "data/spaceinvaders"+ str(i) + ".png"
            # Loading the image
            #input_image = ndimage.imread(input_path, flatten=False, mode=None)
            #print("Loading is done.")
            #print("processing...")
            #segment(input_image, sigma, k, min,i)

            #for i in range(95,100):  #--- for space invaders
            #for i in range(215,218):
        	#input_path = "data/spaceinvaders"+ str(i) + ".png"
            #input_path = "data/new"+ str(i) + ".png"
        	# Loading the image
            #input_image = ndimage.imread(observation, flatten=False, mode=None)
        	

