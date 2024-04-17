import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt
from collections import Counter

style.use('fivethirtyeight')

k=3
distances = []

X1=[[1, 2], [2, 3], [3, 1]]
X2=[[6, 5], [7, 7], [8, 6]]
data = {'k':X1, 'r': X2}
predict = [5, 7]

for group in data:
    print (data)
    for features in data[group]:
        euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
        print(features, euclidean_distance)
        print([euclidean_distance, group])
        distances.append([euclidean_distance, group])
print("distances:")
print(distances)


for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
tmp_distances=[i[1] for i in sorted(distances)]
print(tmp_distances)
tmp_distances=[i[1] for i in sorted(distances)[:k]]
print(tmp_distances)
print(Counter(tmp_distances))
votes = [i[1] for i in sorted(distances)[:k]]

vote_result = Counter(votes).most_common(1)[0][0]
print(vote_result)

[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in data[i]] for i in data]
# same as:
##for i in dataset:
##    for ii in dataset[i]:
##        plt.scatter(ii[0],ii[1],s=100,color=i)

plt.scatter(predict[0], predict[1], s=100)


plt.scatter(predict[0], predict[1], s=100, color=vote_result)
plt.show()


#######################################################

from sklearn.neighbors import K
K=4
X = [[1, 2], [2, 3],[2, 4],[3, 1],[3,3],[4,2],[5,5],[6, 5], [6,6], [7, 7], [8, 6],
[7,5], [8,5]]
y = ['C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C2', 'C2', 'C2', 'C2', 'C2', 'C2', 'C2']
neigh = K(n_neighbors=K)
neigh.fit(X, y)
pred=neigh.predict([[4,4]])
print(pred)