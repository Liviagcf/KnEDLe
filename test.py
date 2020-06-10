list2 = [[0,1], [0,0,2], [3,3,4]]
l = []

for list in list2:
    best = list.pop(list.index(max(list)))

    second_best = list.pop(list.index(max(list)))
    print(best, second_best, best-second_best)
    l.append(best-second_best)

print(l)
