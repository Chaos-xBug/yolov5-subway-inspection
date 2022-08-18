# python3.7.13
# utf-8
# get anchors
# lht


import utils.autoanchor as autoAC
 
# 对数据集重新计算 anchors
new_anchors = autoAC.kmean_anchors('./data/train.yaml', 20, 1088, 4.0, 1000, True)
print(new_anchors)