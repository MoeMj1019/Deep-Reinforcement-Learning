from rl_agent import Agent # own implementations
from rl_environment import GridWorld # own implementations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class Render:
    def __init__(self, env:GridWorld=GridWorld(), agent:Agent=Agent(), textures_map=None , scale=16):
        self.env = env
        self.agent = agent
        # self.fig, self.ax = plt.subplots()
        self.scale = scale
        self.textures_map = textures_map
        # if textures_map: # check if dimensions of and sizes of textures are fitting
            # pass
        
    def start(self):
        plt.ion()
    def end(self):
        plt.ioff()
    def show(self, **kwargs):
        plt.show()

    def run(self, num_episodes, num_steps):
        pass

    # downscale and normelize values of an image
    def normalize_img(self,image:np.ndarray):
        x , y = image.shape[:2]
        return image[::int(x/self.scale),::int(y/self.scale),:] 

    # get a texture from a texture map
    def get_texture(self,x , y, textures):
        return textures[x*self.scale:(x+1)*self.scale,y*self.scale:(y+1)*self.scale,:]

    # load textures and images and downscale them
    def init_rendering_resources(self, configuration=0 , textures_map=None):
        if configuration == 0:
            img = plt.imread('images/rechts.png')
            textures = plt.imread('images/Textures-16.png')
            # pick downscaled and normalized images
            agent_img = self.normalize_img(img) 
            wall_img = self.normalize_img(self.get_texture(4,3,textures))
            shortcut_img = self.normalize_img(self.get_texture(10,12,textures))
            goal_img = self.normalize_img(self.get_texture(11,4,textures))
            negative_goal_img = self.normalize_img(self.get_texture(9,11,textures))
            default_img = self.normalize_img(self.get_texture(6,1,textures))
            # blend the agent image with the deafualt backgraound image  
            # TODO later implement this step while rendering
            agent_img[agent_img == 0] = default_img[agent_img == 0] 
            # object type -> image mapping, size is n of all the nxn images 
            self.textures_map = {'size': self.scale,'agent': agent_img,'wall': wall_img , 'shortcut': shortcut_img,
                    'goal': goal_img, 'negative_goal': negative_goal_img,'default': default_img}
        else:
            if isinstance(textures_map, dict):
                self.textures_map = textures_map
            else:
                print('pick valid configuration, available : [0,] or pass a texture map')

    def axes_settings(self, axes:plt.Axes, title=''): # set the axes settings
            axes.xaxis.set_major_locator(ticker.MultipleLocator(self.scale+1))
            axes.yaxis.set_major_locator(ticker.MultipleLocator(self.scale+1))
            xlabels = axes.get_xticklabels()
            ylabels = axes.get_yticklabels()
            axes.set_xticklabels([int(i.get_position()[0] / (self.scale+1)) for i in xlabels]) 
            axes.set_yticklabels([int(i.get_position()[1] / (self.scale+1)) for i in ylabels]) 
            # ax.grid(axis='both', color='0.95')
            axes.set_title(title)

    def draw_values(self, axes:plt.Axes,text_color='black',**kwargs): # draw values on the grid
        state_values = self.agent.stateValueFunc
        policy = self.agent.policy
        symbols = {'L':'←','U':'↑','R':'→', 'D':'↓','goal':'┼','negative_goal':'─'}

        if isinstance(state_values, dict):
            x_offset = kwargs.get('sv_x_offset', 0.5)
            y_offset = kwargs.get('sv_y_offset', 0.8)   
            for key , value in state_values.items():
                text_kwargs = dict(ha='center', va='center', fontsize=self.scale, color=text_color)
                text = axes.text(key[1]*(self.scale + 1) + (self.scale * x_offset),
                                key[0]*(self.scale + 1) + (self.scale * y_offset),
                                symbols.get(value , value) , text_kwargs)
        if isinstance(policy, dict):
            x_offset = kwargs.get('policy_x_offset', 0.5)
            y_offset = kwargs.get('policy_y_offset', 0.4)
            for key , value in policy.items(): 
                text_kwargs = dict(ha='center', va='center', fontsize=self.scale, color=text_color)
                try:
                    if key in self.env.terminalStates.get('goal', []):
                        text = axes.text(key[1]*(self.scale + 1) + (self.scale * x_offset),
                                    key[0]*(self.scale + 1) + (self.scale * y_offset),
                                    symbols.get('goal', '+') , text_kwargs)
                    elif key in self.env.terminalStates.get('negative_goal', []):
                        text = axes.text(key[1]*(self.scale + 1) + (self.scale * x_offset),
                                    key[0]*(self.scale + 1) + (self.scale * y_offset),
                                    symbols.get('negative_goal', '-') , text_kwargs)
                    else:
                        text = axes.text(key[1]*(self.scale + 1) + (self.scale * x_offset),
                                key[0]*(self.scale + 1) + (self.scale * y_offset),
                                symbols.get(value , value) , text_kwargs)
                except AttributeError:
                    text = axes.text(key[1]*(self.scale + 1) + (self.scale * x_offset),
                                key[0]*(self.scale + 1) + (self.scale * y_offset),
                                symbols.get(value , value) , text_kwargs)
                
    def colorState(self, img, state ,fill = 'black'):
        colors = {'black' : (0,0,0,1),
                    'white' : (1,1,1,1),
                    'grey' : (0.2,0.2,0.2,1),
                    'red' : (1,0,0,1),
                    'green' : (0,1,0,1),
                    'blue' : (0,0,1,1),
                    'yellow' : (1,1,0,1),}
        
        x = state[0]*(self.scale +1)+1
        y = state[1]*(self.scale +1)+1
        if isinstance(fill,np.ndarray) or isinstance(fill,tuple) and len(fill) == img.shape[-1]:
            img[x:x+self.scale,y:y+self.scale,:] = fill
        else:
            img[x:x+self.scale,y:y+self.scale,:] = colors.get(fill, (0,0,0,1))

    def colorMap(self,img):
        # color shortcuts blue
        for state in self.env.terrain.get('shortcut', []):
            self.colorState(img,state,(0,0,1,0.7))

        # color walls grey
        for state in self.env.terrain.get('wall', []):
            self.colorState(img,state,'grey')
        
        # color states based on their values
        state_values = self.agent.stateValueFunc
        if isinstance(state_values, dict) and len(state_values) > 0:
            v_max = max([v**2 for v in state_values.values()])
            v_min = min(state_values.values())

        for state , value in state_values.items():
            if self.env.isValidState(state):
                x = state[0]*(self.scale +1)+1
                y = state[1]*(self.scale +1)+1
                try:
                    v_scaled = (value - v_min) / (v_max - v_min)
                    # f(x) = (x - input_start) / (input_end - input_start) * (output_end - output_start) + output_start
                    # v_scaled = (v_scaled - 0) / (1 - 0) * (1 - 0.2) + 0.2
                    img[x:x+self.scale,y:y+self.scale,:] = (v_scaled,0,v_scaled,1)
                except ZeroDivisionError:
                    img[x:x+self.scale,y:y+self.scale,:] = (1,0,1,1)

    def renderEnv(self,style = 'image', title:str='Gridworld', results=False, **kwargs):
        '''
        with the image mode it's possible to distrain images or pixel art for objects in the environment
        '''               
        # -----------------------------------------
        if style == 'color map':
            # plt.ion()
            
            world_img = np.ones(shape=(self.env.hight * (self.scale+1) +1, self.env.width * (self.scale+1) +1, 4), dtype='float')
            world_img[0::self.scale +1,:,:3] = 0.9 # the +1 for the seperatins
            world_img[:,0::self.scale+1,:3] = 0.9

            self.colorMap(world_img)

            fig , ax = plt.subplots(1,1) 
            ax.imshow(world_img)
            self.axes_settings(ax ,title)
            if results:
                self.draw_values(ax, text_color=(0.9,0.9,0.9))
            plt.draw()
            # plt.show()
        # -----------------------------------------
        if style =='image' :
            # plt.ion()
            # render_image = True
            # if isinstance(self.textures_map, dict):
            #     render_image = True

            world_img = np.ones(shape=(self.env.hight * (self.scale+1) +1, self.env.width * (self.scale+1) +1, 4), dtype='float') # + (hight + 1 and width + 1 ) to count for seperatins
            world_img[0::self.scale +1,:,:3] = 0.90 # the +1 for the seperatins
            world_img[:,0::self.scale+1,:3] = 0.90
            
            for x , y in np.ndindex(self.env.grid.shape):
                state = (x,y)
                if state == self.env.state :
                    try:
                        self.colorState(world_img, (x,y), fill=self.textures_map.get('agent' , 'blue'))
                    except AttributeError:
                        self.colorState(world_img, (x,y), fill='blue')
                elif state in self.env.terminalStates.get('goal',[]):
                    try:
                        self.colorState(world_img, (x,y), fill=self.textures_map.get('goal' , 'green'))
                    except AttributeError:
                        self.colorState(world_img, (x,y), fill='green')
                elif state in self.env.terminalStates.get('negative_goal',[]):
                    try:
                        self.colorState(world_img, (x,y), fill=self.textures_map.get('negative_goal' , 'red'))
                    except AttributeError:
                        self.colorState(world_img, (x,y), fill='red')
                elif state in self.env.terrain.get('wall',[]):
                    try:
                        self.colorState(world_img, (x,y), fill=self.textures_map.get('wall' , 'grey'))
                    except AttributeError:
                        self.colorState(world_img, (x,y), fill='grey')
                elif state in self.env.terrain.get('shortcut',[]):
                    try:
                        self.colorState(world_img, (x,y), fill=self.textures_map.get('shortcut' , 'yellow'))
                    except AttributeError:
                        self.colorState(world_img, (x,y), fill='yellow')
                else:
                    try:
                        self.colorState(world_img, (x,y), fill=self.textures_map.get('default' , 'white'))
                    except AttributeError:
                        self.colorState(world_img, (x,y), fill='white')

            fig , ax = plt.subplots(1,1) 
            ax.imshow(world_img)
            self.axes_settings(ax, title)
            if results:
                self.draw_values(ax)
            plt.draw()
            # plt.show()
        # -----------------------------------------
        if style == 'ascii' :
            for i in range(0, self.env.hight):
                print('-'* 4 * self.env.width )
                out = '| '
                for j in range(0, self.env.width):
                    if self.env.grid[i, j] == 0:
                        token = '0'
                    if self.env.grid[i, j] == 1:
                        token = '0'
                    if self.env.grid[i, j] == self.env.state:
                        token = 'A'
                    if self.env.grid[i, j] == 2:
                        token = '#'
                    out += token + ' | '
                print(out)
            print('-'* 4 * self.env.width )