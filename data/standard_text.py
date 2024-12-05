import numpy as np
from PIL import Image, ImageDraw, ImageFont


class StdText:
    def __init__(self, font_dir, charset):
        self.height = 64
        self.max_width = 1000
        self.border_width = 10
        self.border_height = 4
        self.font_height = self.height - 2 * self.border_height
        self.charset = charset
        base_font = list(map(lambda x: ImageFont.FreeTypeFont(str(x), self.get_valid_height(str(x))), font_dir.glob('*')))
        self.font_dict = {
            'rotate_0': base_font,
            'rotate_90': list(map(lambda x: ImageFont.TransposedFont(x, Image.Transpose.ROTATE_90), base_font)),
            }
        
    def get_valid_height(self, font_path):
        fontsize = 10
        jumpsize = 60
        while True:
            font = ImageFont.FreeTypeFont(font_path, fontsize)
            left, top, right, bottom = font.getbbox(self.charset)
            tmp_height = bottom - top
            if tmp_height < self.font_height - 4:
                fontsize += jumpsize
            else:
                jumpsize = jumpsize // 2
                fontsize -= jumpsize
            if jumpsize <= 1:
                break
        return fontsize

    def draw_text(self, text):
        char_x = self.border_width
        bg = Image.new('RGB', (self.max_width, self.height), color=(127, 127, 127))
        draw = ImageDraw.Draw(bg)
        
        orientation = np.random.choice(list(self.font_dict.keys()))
        for char in text:
            font = np.random.choice(self.font_dict[orientation])
            left, top, right, bottom = font.getbbox(char)
            
            char_x += np.random.randint(-5, 5)
            remain_gap = self.height + top - bottom
            if remain_gap > 2 * self.border_height:
                top_offset_min = self.border_height - top
                top_offset_max = top_offset_min + remain_gap - self.border_height
            elif remain_gap >= 0:
                top_offset_min = -top
                top_offset_max = top_offset_min + remain_gap
            else:
                top_offset_min = remain_gap - top
                top_offset_max = -top
            top_offset_max = max(top_offset_max, top_offset_min + 1) 
            top_offset = np.random.randint(top_offset_min, top_offset_max)
            draw.text((char_x, top_offset), char, fill=(0, 0, 0), font=font, stroke_width=0)
            char_x += right - left
        
        canvas = np.array(bg).astype(np.uint8)
        canvas = canvas[:, :char_x + self.border_width, :]

        return canvas
