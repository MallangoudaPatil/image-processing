from PIL import Image
from pylab import *

im = array(Image.open("D:\\Udemy\\Python Digital Image Processing\\image\\fruits.png"))
imshow(im)
#Select 4 points
pt = ginput(4)

print('You selected : ', pt)

show()
