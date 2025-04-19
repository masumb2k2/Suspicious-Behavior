
import easyocr
from Slogans import  slogans
# Read in images or video
IMAGE_PATH = 'images.png'
#IMAGE_PATH = 'surf.jpeg'

# Reading image
reader = easyocr.Reader(['bn','en'])
result = reader.readtext(IMAGE_PATH)

# Combine all detected text lines into a single sentence
sentence = " ".join([detection[1] for detection in result])
print(sentence)

# Define slogans

# Check if any slogan matches the detected sentence
for slogan in slogans:
    if slogan in sentence:
        print("Yes")
        break
else:
    print("No match found")
