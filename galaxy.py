from fastai.vision import *
import torch
from torch import nn
import numpy as np


eps = 1e-12


task_sectors = {
    1: slice(0, 3),
    2: slice(3, 5),
    3: slice(5, 7),
    4: slice(7, 9),
    5: slice(9, 13),
    6: slice(13, 15),
    7: slice(15, 18),
    8: slice(18, 25),
    9: slice(25, 28),
    10: slice(28, 31),
    11: slice(31, 37),
}


def normalize(q):
    return q / (q.sum(dim=1) + eps)[:, None]


def answer_probability(x):
    # clip probabilities
    nb = x.shape[0]
    x = x.clamp_min(0.)
    
    # normalize
    q1 = normalize(x[:, task_sectors[1]])
    q2 = normalize(x[:, task_sectors[2]])
    q3 = normalize(x[:, task_sectors[3]])
    q4 = normalize(x[:, task_sectors[4]])
    q5 = normalize(x[:, task_sectors[5]])
    q6 = normalize(x[:, task_sectors[6]])
    q7 = normalize(x[:, task_sectors[7]])
    q8 = normalize(x[:, task_sectors[8]])
    q9 = normalize(x[:, task_sectors[9]])
    q10 = normalize(x[:, task_sectors[10]])
    q11 = normalize(x[:, task_sectors[11]])
    
    # reweight 
    w1 = 1.0
    w2 = q1[:, 1] * w1
    w3 = q2[:, 1] * w2
    w4 = w3
    w5 = w4
    w6 = 1.0
    w7 = q1[:, 0] * w1
    w8 = q6[:, 0] * w6
    w9 = q2[:, 0] * w2
    w10 = q4[:, 0] * w4
    w11 = w10
    
    wq1 = w1*q1
    wq2 = w2[:, np.newaxis]*q2
    wq3 = w3[:, np.newaxis]*q3
    wq4 = w4[:, np.newaxis]*q4
    wq5 = w5[:, np.newaxis]*q5
    wq6 = w6*q6
    wq7 = w7[:, np.newaxis]*q7
    wq8 = w8[:, np.newaxis]*q8
    wq9 = w9[:, np.newaxis]*q9
    wq10 = w10[:, np.newaxis]*q10
    wq11 = w11[:, np.newaxis]*q11
    
    return torch.cat([wq1, wq2, wq3, wq4, wq5, wq6, wq7, wq8, wq9, wq10, wq11], dim=1)
    
    
class GalaxyOutput(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return answer_probability(x)
    
    
def resize_one(fn, i, path_hr, path_lr):
    dest = path_lr/fn.relative_to(path_hr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    im = PIL.Image.open(fn)
    #targ_sz = resize_to(img, size, use_min=True)
    #img = img.resize(targ_sz, resample=PIL.Image.BILINEAR).convert('RGB')
    width, height = im.size   # Get dimensions

    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2

    im=im.crop((left, top, right, bottom))
    im.save(dest)

    
def create_cropped_dataset(path, output_path):
    path_hr = path/'images_training_rev1'
    path_crop = output_path/'images_training_rev1_cropped'
    il = ImageList.from_folder(path_hr)
    parallel(partial(resize_one, path_hr=path_hr, path_lr=path_crop), il.items)
    
    
## write the out the question logic as functions
def vec2labels(v):
    """
    Translate the 37D vector into a list of labels
    """
    return task1(v, [])


def task1(v, labels):
    """
    Is the galaxy:
        smooth
        features or disk
        star of artifact
    """
    r = np.argmax(v[task_sectors[1]])
    if r == 0:
        labels.append('smooth')
        return task7(v, labels)
    elif r == 1:
        labels.append('features or disk')
        return task2(v, labels)
    else:
        labels.append('star or artifact')
        return labels
        

def task2(v, labels):
    """
    Could this be disk viewed edge-on?
        yes
        no
    """
    r = np.argmax(v[task_sectors[2]])
    if r == 0:
        labels.append('edge-on disk')
        return task9(v, labels)
    else:
        return task3(v, labels)
    
    
def task3(v, labels):
    """
    Is there are sign of a bar feature through the 
    centre of the galaxy?
        yes
        no
    """
    r = np.argmax(v[task_sectors[3]])
    if r == 0:
        labels.append('bar')
    return task4(v, labels)


def task4(v, labels):
    """
    Is there any sign of a spiral arm
    pattern?
        yes
        no
    """
    r = np.argmax(v[task_sectors[4]])
    if r == 0:
        labels.append('spiral')
        return task10(v, labels)
    else:
        return task5(v, labels)
    

def task5(v, labels):
    """
    How prominent is the central bulge, compared
    with the rest of the galaxy?
        no bulge
        just noticeable
        obvious
        dominant
    """
    r = np.argmax(v[task_sectors[5]])
    if r == 0:
        labels.append('no bulge')
    elif r == 1:
        labels.append('just noticeable bulge')
    elif r == 2:
        labels.append('obvious bulge')
    else:
        labels.append('dominant bulge')
    
    return task6(v, labels)


def task6(v, labels):
    """
    Is there anything else odd?
        yes
        no
    """
    r = np.argmax(v[task_sectors[6]])
    if r == 0:
        return task8(v, labels)
    else:
        return labels
    
    
def task7(v, labels):
    """
    How rounded is it?
        completely round
        in between
        cigar-shaped
    """
    r = np.argmax(v[task_sectors[7]])
    if r == 0:
        labels.append('completely round')
    elif r == 1:
        labels.append('somewhat round')
    else:
        labels.append('cigar shaped')
        
    return task6(v, labels)


def task8(v, labels):
    """
    Is the odd feature of ring or is the galaxy 
    disturbed or irregular?
        ring
        lens or arc
        disturbed
        irregular
        other
        merger
        dust lane
    """
    r = np.argmax(v[task_sectors[8]])
    if r == 0:
        labels.append('ring')
    elif r == 1:
        labels.append('lens or arc')
    elif r == 2:
        labels.append('disturbed')
    elif r == 3:
        labels.append('irregular')
    elif r == 4:
        labels.append('other')
    elif r == 5:
        labels.append('merger')
    else:
        labels.append('dust lane')
        
    return labels


def task9(v, labels):
    """
    Does the galaxy have a bulge at its centre?
    If so what shape?
        round
        boxy
        no bulge
    """
    r = np.argmax(v[task_sectors[9]])
    if r == 0:
        labels.append('rounded bulge')
    elif r == 1:
        labels.append('boxy bulge')
    else:
        labels.append('no bulge')
        
    return task6(v, labels)


def task10(v, labels):
    """
    How tightly wound do the spiral arms appear?
        tight
        medium
        loose
    """
    r = np.argmax(v[task_sectors[10]])
    if r == 0:
        labels.append('tight arms')
    elif r == 1:
        labels.append('medium-tight arms')
    else:
        labels.append('loose arms')
        
    return task11(v, labels)


def task11(v, labels):
    """
    How many spiral arms are there?
        1-4
        more than four
        can't tell
    """
    r = np.argmax(v[task_sectors[11]])
    if r < 4:
        labels.append(f'# spiral arms: {r+1}')
    elif r == 4:
        labels.append('# spiral arms: >4')
    else:
        labels.append('# spiral arms: ??')
        
    return task5(v, labels)

