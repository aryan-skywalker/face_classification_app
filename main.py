#face, gender, race, emotion
import os
import pandas as pd
import cv2
from deepface import DeepFace


#mask
import albumentations as A
import torch
import cv2
import numpy as np

from facemask_detection.pre_trained_models import get_model

model = get_model("tf_efficientnet_b0_ns_2020-07-29")
model.eval();
transform = A.Compose([A.SmallestMaxSize(max_size=256, p=1), A.CenterCrop(height=224, width=224, p=1),A.Normalize(p=1)])


#making pdf
import time
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch



#capturing from webcam
import os.path

person_name = input("Enter your name : ")

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "faces/" + person_name + ".png"
        img_onlyname = person_name + ".png"
        if(os.path.isfile(img_name) == False):
            cv2.imwrite(img_name, frame)
            break
        img_name = "faces/" + person_name + str(img_counter) + ".png"
        img_onlyname = person_name + str(img_counter) + ".png"
        #chack if that person name is already present
        while(os.path.isfile(img_name)):
            img_counter += 1
            img_name = "faces/" + person_name + str(img_counter) + ".png"
            img_onlyname = person_name + str(img_counter) + ".png"
        cv2.imwrite(img_name, frame)
        
        
        print("{} written!".format(img_name))
        break

cam.release()

cv2.destroyAllWindows()



#image_taylor = cv2.cvtColor(cv2.imread("kriti.jpeg"), cv2.COLOR_BGR2RGB)
#transformed_image = transform(image=image_taylor)['image']

#input = torch.from_numpy(np.transpose(transformed_image, (2, 0, 1))).unsqueeze(0)

#img = cv2.imread("faces/kriti.jpeg")

data = {
    "Name":[],
    "Age":[],
    "Gender":[],
    "Race":[],
    "Emotion":[],
    "Mask":[]
}


#for all the persons

# for file in os.listdir("faces"):
#     result = DeepFace.analyze(cv2.imread(f"faces/{file}"),actions=("age","gender","race","emotion"))
#     data["Name"].append(file.split(".")[0])
#     data["Age"].append(result[0]["age"])
#     data["Gender"].append(result[0]["dominant_gender"])
#     data["Race"].append(result[0]["dominant_race"])
#     data["Emotion"].append(result[0]["dominant_emotion"])
    
#     image_taylor = cv2.cvtColor(cv2.imread(f"faces/{file}"), cv2.COLOR_BGR2RGB)
#     transformed_image = transform(image=image_taylor)['image']
#     input = torch.from_numpy(np.transpose(transformed_image, (2, 0, 1))).unsqueeze(0)
#     if model(input)[0].item() > 0.5:
#         mask_val = 'Yes'
#     else:
#         mask_val = 'No'
#     data["Mask"].append(mask_val)



#for captured person

file = img_onlyname
result = DeepFace.analyze(cv2.imread(f"faces/{file}"),actions=("age","gender","race","emotion"))
data["Name"].append(file.split(".")[0])
data["Age"].append(result[0]["age"])
data["Gender"].append(result[0]["dominant_gender"])
data["Race"].append(result[0]["dominant_race"])
data["Emotion"].append(result[0]["dominant_emotion"])
    
image_taylor = cv2.cvtColor(cv2.imread(f"faces/{file}"), cv2.COLOR_BGR2RGB)
transformed_image = transform(image=image_taylor)['image']
input = torch.from_numpy(np.transpose(transformed_image, (2, 0, 1))).unsqueeze(0)
if model(input)[0].item() > 0.5:
    mask_val = 'Yes'
else:
    mask_val = 'No'
data["Mask"].append(mask_val)





df = pd.DataFrame(data)
print(df)
#print("Probability of the mask on the face = ", model(input)[0].item())

df.to_csv("people.csv")




#making pdf
full_name = file.split(".")[0]
doc = SimpleDocTemplate("pdf_files/" + str(full_name) +".pdf",pagesize=letter,rightMargin=72,leftMargin=72,topMargin=72,bottomMargin=18)
Story=[]
logo = img_name
formatted_time = time.ctime()

im = Image(logo, 4*inch, 4*inch)
Story.append(im)
Story.append(Spacer(1, 12))
styles=getSampleStyleSheet()

styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
ptext = 'NAME  --->  %s' % full_name
Story.append(Paragraph(ptext, styles["Normal"]))
Story.append(Spacer(1, 12))

ptext = "AGE  --->  " + str(result[0]["age"])
Story.append(Paragraph(ptext, styles["Normal"]))
Story.append(Spacer(1, 12))

ptext = "GENDER  --->  " +  str(result[0]["dominant_gender"])
Story.append(Paragraph(ptext, styles["Normal"]))
Story.append(Spacer(1, 12))

ptext = "RACE  --->  " +  str(result[0]["dominant_race"])
Story.append(Paragraph(ptext, styles["Normal"]))
Story.append(Spacer(1, 12))

ptext = "EMOTION  --->  " +  str(result[0]["dominant_emotion"])
Story.append(Paragraph(ptext, styles["Normal"]))
Story.append(Spacer(1, 12))

ptext = "Wore mask ?  --->  " + str(mask_val)
Story.append(Paragraph(ptext, styles["Normal"]))
Story.append(Spacer(1, 12))

ptext = 'DATE and TIME  --->  %s' % formatted_time
Story.append(Paragraph(ptext, styles["Normal"]))  
doc.build(Story)