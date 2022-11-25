import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Non Maximum Suppression

# BBoxes of the form (x1,y1,x2,y2,c), where (x1,y1) and (x2,y2) are the ends of the BBox and c
# overlap threshold IoU thresh_iou

def nms_pytorch(P : torch.tensor ,thresh_iou : float):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the image 
            along with the class predscores, Shape: [num_boxes,5].
        thresh_iou: (float) The overlap thresh for suppressing unnecessary boxes.
    Returns:
        A list of filtered boxes, Shape: [ , 5]
    """
 
    # we extract coordinates for every 
    # prediction box present in P
    x1 = P[:, 0]
    y1 = P[:, 1]
    x2 = P[:, 2]
    y2 = P[:, 3]
 
    # we extract the confidence scores as well
    scores = P[:, 4]
 
    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)
     
    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()
 
    # initialise an empty list for 
    # filtered prediction boxes
    keep = []
     
 
    while len(order) > 0:
         
        # extract the index of the 
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]
 
        # push S in filtered predictions list
        keep.append(P[idx])
 
        # remove S from P
        order = order[:-1]
 
        # sanity check
        if len(order) == 0:
            break
         
        # select coordinates of BBoxes according to 
        # the indices in order
        xx1 = torch.index_select(x1,dim = 0, index = order)
        xx2 = torch.index_select(x2,dim = 0, index = order)
        yy1 = torch.index_select(y1,dim = 0, index = order)
        yy2 = torch.index_select(y2,dim = 0, index = order)
 
        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])
 
        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1
         
        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
 
        # find the intersection area
        inter = w*h
 
        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim = 0, index = order) 
 
        # find the union of every prediction T in P
        # with the prediction S
        # Note that areas[idx] represents area of S
        union = (rem_areas - inter) + areas[idx]
         
        # find the IoU of every prediction in P with S
        IoU = inter / union
 
        # keep the boxes with IoU less than thresh_iou
        mask = IoU < thresh_iou
        order = order[mask]
    return keep

"""
# Let P be the following
P = torch.tensor([
    [1, 1, 3, 3, 0.95],
    [1, 1, 3, 4, 0.93],
    [1, 0.9, 3.6, 3, 0.98],
    [1, 0.9, 3.5, 3, 0.97]
])

print(float(P[0][0]))

def plot_boxes(tensor):
    #define Matplotlib figure and axis
    fig, ax = plt.subplots()

    #create simple line plot
    ax.plot([0, 10],[0, 10])

    for vector in tensor:
        ax.add_patch(Rectangle((float(vector[0]), float(vector[1])), abs(float(vector[2]) - float(vector[0])), abs(float(vector[3]) - float(vector[1])), edgecolor="red")).set_fill(False)


    #display plot
    plt.show()

filtered_boxes = nms_pytorch(P, 0.8)
print(filtered_boxes)
plot_boxes(filtered_boxes)
"""