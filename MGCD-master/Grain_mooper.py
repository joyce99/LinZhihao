import copy


dataset = 'data/mooper/'
with open(dataset + 'grain.txt') as f:
    f.readline()
    en, cn, sn, dn = f.readline().split(',')
    en, cn, sn, dn = int(en), int(cn), int(sn), int(dn)
with open(dataset+'challenge_exercise.txt', 'r') as f:  # challenge exercise
    dict_1e = {}
    for line in f.readlines():
        line = line.replace('\n', '').split('\t')
        left = int(line[0])
        right = int(line[1]) - 4550
        if left not in dict_1e.keys():
            temp = []
            temp.append(right)
            dict_1e.update({left: copy.copy(temp)})
        else:
            temp = dict_1e[left]
            if right not in temp:
                temp.append(right)
                dict_1e.update({left: copy.copy(temp)})
with open(dataset+'exercise_course.txt', 'r') as f:  # exercise course
    dict_ec = {}
    for line in f.readlines():
        line = line.replace('\n', '').split('\t')
        left = int(line[0])
        right = int(line[1]) - 4550
        if left not in dict_ec.keys():
            temp = []
            temp.append(right)
            dict_ec.update({left: copy.copy(temp)})
        else:
            temp = dict_ec[left]
            if right not in temp:
                temp.append(right)
                dict_ec.update({left: copy.copy(temp)})
with open(dataset+'exercise_subdiscipline.txt', 'r') as f:  # exercise subdiscipline
    dict_es = {}
    for line in f.readlines():
        line = line.replace('\n', '').split('\t')
        left = int(line[0])
        right = int(line[1]) - 4550
        if left not in dict_es.keys():
            temp = []
            temp.append(right)
            dict_es.update({left: copy.copy(temp)})
        else:
            temp = dict_es[left]
            if right not in temp:
                temp.append(right)
                dict_es.update({left: copy.copy(temp)})
with open(dataset+'challenge_discipline.txt', 'r') as f:  # challenge discipline
    dict_1d = {}
    for line in f.readlines():
        line = line.replace('\n', '').split('\t')
        left = int(line[0])
        right = int(line[1]) - 4550
        if left not in dict_1d.keys():
            temp = []
            temp.append(right)
            dict_1d.update({left: copy.copy(temp)})
        else:
            temp = dict_1d[left]
            if right not in temp:
                temp.append(right)
                dict_1d.update({left: copy.copy(temp)})


def equal2(challenge):
    output_kn2 = []
    if challenge in dict_1e:
        exercise = dict_1e[int(challenge)]
        for e in exercise:
            output_kn2.append(e)
            if int(e) in dict_ec:
                course = dict_ec[int(e)]
                for c in course:
                    output_kn2.append(c)
            if int(e) in dict_es:
                subdiscipline = dict_es[int(e)]
                for s in subdiscipline:
                    output_kn2.append(s)
    if challenge in dict_1d:
        discipline = dict_1d[int(challenge)]
        for d in discipline:
            output_kn2.append(d)
    return output_kn2

