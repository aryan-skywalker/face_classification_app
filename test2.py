import time
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
doc = SimpleDocTemplate("form_letter.pdf",pagesize=letter,rightMargin=72,leftMargin=72,topMargin=72,bottomMargin=18)
Story=[]
logo = "faces/kriti.jpeg"
formatted_time = time.ctime()
full_name = "kriti"
im = Image(logo, 4*inch, 4*inch)
Story.append(im)
Story.append(Spacer(1, 12))
styles=getSampleStyleSheet()

styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
ptext = 'NAME ---> %s' % full_name
Story.append(Paragraph(ptext, styles["Normal"]))
Story.append(Spacer(1, 12))

ptext = 'AGE ---> %s' % full_name
Story.append(Paragraph(ptext, styles["Normal"]))
Story.append(Spacer(1, 12))

ptext = 'GENDER ---> %s' % full_name
Story.append(Paragraph(ptext, styles["Normal"]))
Story.append(Spacer(1, 12))

ptext = 'RACE ---> %s' % full_name
Story.append(Paragraph(ptext, styles["Normal"]))
Story.append(Spacer(1, 12))

ptext = 'EMOTION ---> %s' % full_name
Story.append(Paragraph(ptext, styles["Normal"]))
Story.append(Spacer(1, 12))

ptext = 'Wore mask ?  --->  %s' % full_name
Story.append(Paragraph(ptext, styles["Normal"]))
Story.append(Spacer(1, 12))

ptext = 'DATE and TIME  --->  %s' % formatted_time
Story.append(Paragraph(ptext, styles["Normal"]))  
doc.build(Story)