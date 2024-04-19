import nltk
import re
import streamlit as st
import pickle

nltk.download('punkt')
nltk.download('stopwords')

#loading models

clf = pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

#cleaner function
def CleanResume(txt):
    cleanTxt = re.sub('http\S+\s',' ',txt) #remove any links
    cleanTxt = re.sub('RT|cc','',cleanTxt) #RT and cc are not relevant
    cleanTxt = re.sub('@\S+',' ',cleanTxt) #http and attached words
    cleanTxt = re.sub('#\S+',' ',cleanTxt) #remove #words
    cleanTxt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""),' ',cleanTxt)
    cleanTxt = re.sub(r'[^\x00-\x7f]',' ',cleanTxt)
    cleanTxt = re.sub('\s+',' ',cleanTxt)
    
    
   
    
    return cleanTxt



#web app
def main():
    st.title("Resume Screening Application")
    uploaded_file = st.file_uploader('Upload Resume/CV', type = ['txt','pdf','docx'])
    
    if uploaded_file is not None:            #if uploaded file is not blank move into try block
        try:
            resume_bytes = uploaded_file.read()  #read the file
            resume_text = resume_bytes.decode('utf-8') #decode the file
        except:
            #If UTF-8 fails you can try Latin-1
            resume_text = resume_bytes.decode('latin-1')  
        
        cleaned_resume = CleanResume(resume_text)
        
        #st.write(cleaned_resume)
        
        input_features = tfidf.transform([cleaned_resume]) #convert text to sparse array because my model is trained on that
        
        prediction_id = clf.predict(input_features)[0]
        
        #st.write(input_features)
        
        category_mapping = {
                            15:"Java Developer",
                            23:"Testing",
                            6:"Data Science",
                            8:"DevOps Engineer",
                            12:"HR",
                            13:"Hadoop",
                            3:"Blockchain",
                            10:"ETL Developer",
                            18:"Operation Manager",
                            22:"Sales Professional",
                            16:"Mechanical Engineer",
                            1:"Arts",
                            7:"DBA",
                            11:"Electrical Engineer",
                            14:"Health and Fitness",
                            19:"Product Manager",
                            4:"Business Analyst",
                            9:"DotNet Developer",
                            2:'Automation Testing',
                            17:"Network Security Engineer",
                            21:"SAP Developer",
                            5:"Civil Engineer",
                            0:"Advocate",
                            23:"Testing",
                            20:"Python Developer",
                            24:"Web Designer"}
        
        category_name = category_mapping.get(prediction_id,"Unknown")
        st.write("Predicted Category:",category_name)
        
    



if __name__ == '__main__':
    main()


