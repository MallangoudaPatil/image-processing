from PIL import Image

img = Image.open('images/profile.jpg')

#Resize
smaller_img = img.resize((150, 150))
#smaller_img.show()

#Crop
box = (100, 100, 400, 400)
region = img.crop(box)

#region.show()
#Rotation
rotated_img = img.rotate(45)
rotated_img.show()
