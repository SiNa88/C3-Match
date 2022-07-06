import networkx as nx
from operator import itemgetter

# ****************************************************************************************************

# FUNCIONES PARA LA PARTE DE TRANSITIVE CLOSURE

# ****************************************************************************************************
verbose_log = False

def normalizeSmallerSetsInSameLevel(transitivesClosures):
    for n in transitivesClosures:
        tmpList = list(transitivesClosures[n])
        for i in range(0, len(tmpList)):
            for j in range(i + 1, len(tmpList)):
                if (tmpList[i] & tmpList[j]) == tmpList[i]:
                    transitivesClosures[n].remove(tmpList[i])


def createSetFromSetOfSets(setofsets):
    finalset = set()
    for i in setofsets:
        finalset = finalset | set(i)

    return finalset


def normalizeIncludePrevious(transitivesClosures):
    previous = transitivesClosures[0]

    for n in transitivesClosures:
        if verbose_log:
            print("level")
            print(n)
        toInclude = set()
        for i in transitivesClosures[n]:
            current = createSetFromSetOfSets(transitivesClosures[n])
            for j in previous:
                if len(j & current) == 0:
                    if verbose_log:
                        print("individual")
                        print(j)
                    toInclude.add(j)
        if verbose_log:
            print("final")
            print(toInclude)
        transitivesClosures[n] = transitivesClosures[n] | toInclude
        previous = transitivesClosures[n]


def getTransitiveClosures(source, app_, transitivesClosures, cycles_, level):
    #    if level > 10:
    #        import sys
    #        sys.exit("Error message")

    if verbose_log:
        print(source)
    #    raw_input()
    neighbords_ = list(app_.neighbors(source))
    if not level in transitivesClosures:
        transitivesClosures[level] = set()

    descendantsOfNeighbords = set()
    for i in neighbords_:
        descendantsOfNeighbords = descendantsOfNeighbords | set(nx.descendants(app_, i))
        #print(descendantsOfNeighbords)

    #
    #    tmp = set()
    #    tmp.add(source)
    #    transitivesClosures[level].add(frozenset(tmp))

    tmp = set(nx.descendants(app_, source))
    tmp.add(source)
    tmp = frozenset(tmp)
    #    print "tmp"
    #print(tmp)
    #    print "cycles_"
    #    print cycles_
    #    print "transtivesClosures"
    #    print transitivesClosures
    #    print "level"
    #    print level
    if not tmp in transitivesClosures[level]:
        if verbose_log:
            print(tmp)
        transitivesClosures[level].add(tmp)
        if not tmp in cycles_:

            if len(neighbords_) > 0:
                if not (level + 1) in transitivesClosures:
                    transitivesClosures[level + 1] = set()

                tmp = set()
                tmp.add(source)
                transitivesClosures[level + 1].add(frozenset(tmp))

            for n in neighbords_:
                # if not n in descendantsOfNeighbords:

                getTransitiveClosures(n, app_, transitivesClosures, cycles_, level + 1)
    #            tmp=set(nx.descendants(app_,n))
    #            tmp.add(n)
    #            tmp = frozenset(tmp)
    #            transitivesClosures[1].add(tmp)


def transitiveClosureCalculation(source, app_):
    transitivesClosures = {}

    cycles_ = set()

    for i in nx.simple_cycles(app_):
        if verbose_log:
            print(i)
        tmp = frozenset(i)
        if verbose_log:
            print(tmp)
        cycles_.add(tmp)

    getTransitiveClosures(source, app_, transitivesClosures, cycles_, 0)

    normalizeSmallerSetsInSameLevel(transitivesClosures)

    normalizeIncludePrevious(transitivesClosures)

    return transitivesClosures

########################################################################################################################





def longest_simple_paths(graph, source, target):
    longest_paths = []
    longest_path_length = 0
    for path in nx.all_simple_paths(graph, source=source, target=target):
        if len(path) > longest_path_length:
            longest_path_length = len(path)
            longest_paths.clear()
            longest_paths.append(path)
        elif len(path) == longest_path_length:
            longest_paths.append(path)
    return longest_paths

#[encode_20000-0,frame_20000-0,hightrain-0,highinference-0,package-0]
APPS=[]
APP = nx.DiGraph()
APP.add_edges_from([("src","m0"),("src","m1"),("m0","m2"),("m1","m3"),("m2","m4"),("m3","m4")])
#APP.add_edges_from([("src","encode"),("encode","frame"),("frame","lowtrain"),("lowtrain","lowinference"),("frame","hightrain"),("hightrain","highinference"),("lowinference","package"),("highinference","package"),("hightrain","package"),("lowtrain","package")])
APPS.append(APP)
'''APP=nx.DiGraph()
APP.add_edges_from([("src","extract"),("extract", "vectorize"), ("vectorize", "lowtrain"), ("vectorize", "hightrain"), ("lowtrain", "score"),("hightrain", "score")])
APPS.append(APP)'''

for APP in APPS:
    print("****************** Calculating level for each microservice before transitive reduction *****************************")
    for task in APP.nodes():
        #print((task))
        if (task=="src"):
            #print(0)
            continue
        elif (task=="snk"):
            break
        else:
            longest_paths = longest_simple_paths(APP, source="src", target=task)
            #print((longest_paths))
            print("from src to ", task," ",len(longest_paths[0])-1)
    topologicorder_ = list(nx.topological_sort(APP))
    source = topologicorder_[0]
    appsCommunities = list()

    #calculating transitive closures
    appsCommunities.append(transitiveClosureCalculation(source, APP))
    transitiveClosureCalculation(source, APP)
    listOfTransitiveClosures=[]
    for appCommu in appsCommunities[0].items():
        listOfTransitiveClosures .append(appCommu[1])


    print(listOfTransitiveClosures)

    #Calculating level for each community of microservices based on transitive closures
    print("****************** Microservices community based on transitive reduction *****************************")
    app_level=[]
    for appCommu in listOfTransitiveClosures[3]:
        print(appCommu)
        level = []
        for microservice in appCommu:
            if (microservice=="src"):
                #print(0)
                continue
            else:
                longest_paths = longest_simple_paths(APP, source="src", target=microservice)
                level.append(len(longest_paths[0])-1)
                #print((longest_paths))
        if (microservice != "src"):
            #print("from src to ", appCommu," ",min(level))
            app_level.append({'level':min(level), 'appComm': appCommu})

    sorted_app_level = sorted(app_level, key=itemgetter('level'))
    print("****************** Calculating level for each community of microservices based on transitive reduction *****************************")
    for i in range(len(sorted_app_level)):
        print("from src to ", sorted_app_level[i]['appComm']," ",sorted_app_level[i]['level'])
    print()
    print()
    #print("****************** Second case study *****************************")