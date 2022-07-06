from matching import Player
from matching.games import HospitalResident
import sys
import urllib
import yaml
from yaml import load, dump, Loader
from io import StringIO

'''
url = "https://zenodo.org/record/3688091/files/residents.yml"
with urllib.request.urlopen(url) as response:
    resident_preferences = yaml.full_load(response.read())


url = "https://zenodo.org/record/3688091/files/hospitals.yml"
with urllib.request.urlopen(url) as response:
    hospital_preferences = yaml.full_load(response.read())


url = "https://zenodo.org/record/3688091/files/capacities.yml"
with urllib.request.urlopen(url) as response:
    hospital_capacities = yaml.full_load(response.read())
'''

resident__ = open('D:\\00Research\\matching\\scheduler\\TO-Upload\\c3-match-main\\real-experiments\\dataflow-load\\MPL.yaml', 'r')
resident_preferences = yaml.full_load(resident__)
resident__.close()

hospital__ = open('D:\\00Research\\matching\\scheduler\\TO-Upload\\c3-match-main\\real-experiments\\dataflow-load\\DPL.yaml', 'r')
hospital_preferences = yaml.full_load(hospital__)
hospital__.close()

hospital_cap = open('D:\\00Research\\matching\\scheduler\\TO-Upload\\c3-match-main\\real-experiments\\dataflow-load\\capacities-testbed.yml', 'r')
hospital_capacities = yaml.full_load(hospital_cap)
hospital_cap.close()

#print(len(resident_preferences), len(hospital_preferences), sum(hospital_capacities.values()))


game = HospitalResident.create_from_dictionaries(
    resident_preferences, hospital_preferences, hospital_capacities
)

mymatching = game.solve(optimal="resident")
#print(type(mymatching))

old_stdout = sys.stdout
 
# This variable will store everything that is sent to the standard output
 
result = StringIO()
 
sys.stdout = result
 
# Here we can call anything we like, like external modules, and everything that they will send to standard output will be stored on "result"
 
#do_fancy_stuff()
print((mymatching))


# Redirect again the std output to screen
 
sys.stdout = old_stdout

data = result.getvalue()
data = data.replace("\n","")
#print (data)
#str = "key1=value1;key2=value2;key3=value3"
#dict_file = [
#	{'sports' : ['soccer', 'football', 'basketball', 'cricket', 'hockey', 'table tennis']},
#	{'countries' : ['Pakistan', 'USA', 'India', 'China', 'Germany', 'France', 'Spain']}
#	]

#{vm-aws: [highAccuracy], vm-exo: [analysis], t-1: [transcoding, packaging], e-0: [snk, lowAccuracy], e-1: [src], e-2: [framing]}
data = data.replace(" ","")
data = data.replace("{","[{") 
data = data.replace("}","}]") 
data = data.replace("],","]},{") 
#print(data)
#[{vm-aws:[highAccuracy]},{vm-exo:[analysis]},{t-1:[transcoding,packaging]},{e-0:[snk,lowAccuracy]},{e-1:[src]},{e-2:[framing]}]
data = data.replace("[{","[{\"")
data = data.replace(",","\",\"") 
data = data.replace("]}","\"]}") 
data = data.replace("]}\",\"{","]},{\"") 
data = data.replace(":[","\":[\"") 

print(data)
#dicttttt = dict(x.split(":") for x in data.split("],"))
#print(type(dicttttt))

#fileObject = open("matching-testbed.yaml",'w').close()
fileObject = open(r"D:\\00Research\\matching\\scheduler\\TO-Upload\\c3-match-main\\real-experiments\\dataflow-load\\matching-testbed.yaml",'w')
yaml.dump((data),fileObject)
fileObject.close()

assert game.check_validity()
assert game.check_stability()

matched_residents = []
for _, residents in mymatching.items():
    for resident in residents:
        matched_residents.append(resident.name)

unmatched_residents = set(resident_preferences.keys()) - set(matched_residents)
#print(unmatched_residents)