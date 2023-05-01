import numpy as np
import matplotlib.pyplot as plt

def normalize_img(image, size=16):
    x , y = image.shape[:2]
    return image[::int(x/size),::int(y/size),:] # downscale and normelize values 

def get_texture(x,y ,textures , size):
    return textures[x*size:(x+1)*size,y*size:(y+1)*size,:]

def init_rendering_resources(configuration=0):
    texture_map = {}
    if configuration == 0:
        scaler = 16 # normalized img size
        img = plt.imread('images/rechts.png')
        textures = plt.imread('images/Textures-16.png')
        # pick downscaled and normalized images
        agent_img = normalize_img(img,scaler) 
        wall_img = normalize_img(get_texture(4,3,textures, scaler), scaler)
        shortcut_img = normalize_img(get_texture(10,12,textures, scaler), scaler)
        goal_img = normalize_img(get_texture(11,4,textures, scaler), scaler)
        negative_goal_img = normalize_img(get_texture(9,11,textures, scaler), scaler)
        default_img = normalize_img(get_texture(6,1,textures, scaler), scaler)

        agent_img[agent_img == 0] = default_img[agent_img == 0] 
        # object type -> image mapping, size is n of all the nxn images 
        texture_map = {'size': 16,'agent': agent_img,'wall': wall_img , 'shortcut': shortcut_img,
                'goal': goal_img, 'negative_goal': negative_goal_img,'default': default_img}
        # plt.imshow(wall_img)
    else:
        print('pick valid configuration, available : [0,]')

    return texture_map