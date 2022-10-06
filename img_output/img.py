"""
Source: https://stackoverflow.com/a/65695715
"""
import sys
import math
from PIL import Image

numcols = int(sys.argv[1])
numimages = len(sys.argv) - 2  # Ignore name of program and column input
numrows = math.ceil(numimages / numcols)  # Number of rows

sampleimage = Image.open(sys.argv[2])  # Open first image, just to get dimensions
width, height = sampleimage.size  # PIL uses (x, y)

outimg = Image.new(
    "RGBA", (numcols * width, numrows * height), (0, 0, 0, 0)
)  # Initialize to transparent

# Write to output image. This approach copies pixels from the source image
for i in range(numimages):
    currimage = Image.open(sys.argv[2 + i])
    for j in range(width):
        for k in range(height):
            currimgpixel = currimage.getpixel((j, k))
            outimg.putpixel(
                ((i % numcols * width) + j, (i // numcols * height) + k), currimgpixel
            )
outimg.save("output.png")
