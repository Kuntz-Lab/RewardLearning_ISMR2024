#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg
from matplotlib.widgets import RadioButtons

''' Tutorials:
https://stackoverflow.com/questions/34975972/how-can-i-make-a-video-from-array-of-images-in-matplotlib
https://matplotlib.org/2.1.2/gallery/animation/dynamic_image2.html


'''
# imgplot = plt.imshow(img)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

# img = [img] # some array of images
frames = [] # for storing the generated images
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2) 

for i in range(30):
    img = mpimg.imread(f'/home/dvrk/shape_servo_data/generalization/surgical_setup/plane_vis/1/image{i:03}.png')
    # f, axarr = plt.subplots(2,2)
    # img1 = axarr[0,0].imshow(img)
    # img2 = axarr[0,1].imshow(img)


    img1 = ax1.imshow(img)
    img2 = ax2.imshow(img)
    # frames.append([fig])

    frames.append([img1, img2])
    # frames.append([plt.imshow(img, cmap=cm.Greys_r,animated=True)])

ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True,
                                repeat_delay=1000)
# ani.save('movie.mp4')

# adjust radio buttons
axcolor = 'lightgoldenrodyellow'
rax = plt.axes([0.05, 0.4, 0.15, 0.30],
               facecolor=axcolor)
   
radio = RadioButtons(rax, ['red', 'blue', 'green'],
                     [True,False,False,False],
                     activecolor='r')
 
def color(labels):
    l.set_color(labels)
    fig.canvas.draw()
radio.on_clicked(color)

plt.show()