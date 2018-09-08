import difflib
import generateDoc

def postProcessing(words):

    file = open("words.txt", "r")
    lineList = file.readlines()
    for i in range(0, len(lineList)):
        lineList[i]=lineList[i].replace("\n","")

    #print(lineList)


    result=[]
    tempName="none"

    for word in words:
        print("in post file")
        maxRatio = 0.0
        for i in range(0, len(lineList)):
            ratio = difflib.SequenceMatcher(None,word, lineList[i]).ratio()
            if ratio >= 0.5 and len(word) >= 1:
                # print("matched with : " + test.name)
                if maxRatio < ratio:
                    #print(maxRatio)
                    tempName = lineList[i]
                    maxRatio = ratio
                    #print(maxRatio)
        result.append(tempName)
    return result


if __name__ =='__main__':
    generateDoc.generate(postProcessing(["mothe","luve","hellu"]))