
import os
import subprocess
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fontman
import textwrap

class Text2Image(object):
    '''
    convert text to image
    '''
    def __init__(self, config):    
        self.font_size_range = config['FONT_SIZE_RANGE']
        self.line_width_range = config['LINE_WILD_RANGE']
        font_paths = os.listdir("./ttfs")
        self.font_paths = [os.path.join("./ttfs", font_path) for font_path in font_paths if "ttf" in font_path]

    def convert(self, text, is_train=True):    
        # NOTE: we do some trim to the original text to ensure the canvas not to be too big
        text = text[:77*7]
        font_path = random.choice(self.font_paths)

        if is_train:
            font = ImageFont.truetype(font_path, random.choice(range(self.font_size_range[0], self.font_size_range[1])))
            line_width = random.choice(range(self.line_width_range[0], self.line_width_range[1]))
        else:
            font = ImageFont.truetype(font_path, (self.font_size_range[0]+self.font_size_range[1]) // 2)
            line_width = (self.line_width_range[0]+self.line_width_range[1]) // 2
        
        # text_color = np.uint8(255*np.random.rand(3))
        text_color = np.uint8([20, 20, 20])

        # import pdb; pdb.set_trace()
        font_width, font_height = font.getsize(text)
        line_width = int(np.sqrt(len(text)))

        texts = textwrap.fill(text, line_width)

        max_width = 0; max_height = 0
        for text_line in texts.split('\n'):
            font_width, font_height = font.getsize(text_line)
            max_width = max(max_width, font_width)
            max_height += font_height

        # generate a canvas image with random color
        # random_color = np.uint8(255*np.random.rand(3))
        random_color = np.uint8([200, 200, 200])
        max_size = max(max_height, max_width)
        canvas_size = int(max_size * 1.2)

        img = Image.new(mode="RGB", size=(canvas_size, canvas_size), color=tuple(random_color.tolist()))
        canvas = ImageDraw.Draw(img)

        canvas.text(((canvas_size - max_width) // 2, (canvas_size - max_height) // 2), texts, fill=tuple(text_color.tolist()), font=font)

        return img
