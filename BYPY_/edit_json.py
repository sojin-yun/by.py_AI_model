"""
import os
import json

os.chdir('C:/Users/MEDICAL IT/Downloads/화재 발생 예측 영상/Validation/label_SnF')

for fp in os.listdir('.'):
    with open(fp, "r", newline='\n', encoding="utf-8-sig") as f:
        contents = f.read() # string으로 읽어
        json_data = json.loads(contents)
        for i in range(len(json_data['annotations'])-1,-1,-1):
            data = json_data['annotations'][i]['class']
            if data!='01' and data!='02' and data!='03' and data!='04':
                del json_data['annotations'][i]
                print(f"class {data} is deleted!")
        f.close()
        
        with open(fp, 'w') as file:
            json.dump(json_data, file, indent=4)
            file.close()
            print(fp , "is completed!")

"""

'''

with open("bb.json", "r", encoding="utf8") as f:
    contents = f.read() # string으로 읽어
    json_data = json.loads(contents) #loads하면 dict type으로 변함
    #print(json_data['annotations'][0])
    #print(json_data['annotations'][1])
    #print(len(json_data['annotations']))
    for i in range(len(json_data['annotations'])):
        data = json_data['annotations'][i]['class']
        if data!='02' and data!='01':
            del json_data['annotations'][i]
            print(f"class {data} is deleted!")
    
    print(json_data)

'''

"""
os.chdir('C:/Users/MEDICAL IT/Desktop/test')

for fp in os.listdir('.'):
    #print(fp)

    if fp.endswith('.json'):
        #.json 파일만 열기 
        f = open(fp, 'r+')
        content = f.read()
        f.close()

        if 
        text = content.replace('image(','image_').replace(')','')
        #print(text)

        #수정할 파일을 다시 열어서 덮어쓴다.
        with open(fp, 'w') as file:
            file.write(text)
            file.close()
            print(fp , "완료")
"""

import os
import json
from collections import OrderedDict

file_data = OrderedDict()

with open("tmpjjson.json", "rt", newline='', encoding="utf-8-sig") as f:
    contents = f.read() 
    json_data = json.loads(contents)
    
    key_annot = json_data["annotations"]
    for i in range(len(key_annot)):
        if "polygon" in key_annot[i]: 
            key_poly = key_annot[i]["polygon"]
            a=[]
            b=[]
            for j in key_poly:
                a.append(j[0])
                b.append(j[1])
            print(max(a),min(a),max(b),min(b))
            #segmentation = [item for l in key_poly for item in l]

    f.close()

