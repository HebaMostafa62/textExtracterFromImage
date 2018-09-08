#from docx import Document

def generate(list ,fileName):
    #print("writting file")
    file = open(fileName,'w')
    file.write(list)
    file.close
    # document = Document()
    # document.add_heading('Document Title', 0)
    # p = document.add_paragraph('--')
    #
    #
    # for word in list:
    #     p.add_run(" "+word)
    #
    #
    # document.save('output.docx')



