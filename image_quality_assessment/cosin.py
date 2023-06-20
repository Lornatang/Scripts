from PIL import Image
from numpy import average, linalg, dot
 
 
def image_similarity_vectors_via_numpy(image1, image2):
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    res = dot(a / a_norm, b / b_norm)
    return res
 
 
image1 = Image.open("sr_071911.bmp")
image2 = Image.open("hr_071911.bmp")
cosin = image_similarity_vectors_via_numpy(image1, image2)
 
print(cosin)