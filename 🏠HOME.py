import streamlit as st

one,two=st.columns((9,2))

one.title("DISEASE PREDICTION APP!")
two.image("https://cdn2.iconfinder.com/data/icons/medical-2215/64/clip_board-pen-notes-document-medical-hospital-512.png",width=100)


st.markdown(""" <br> """,True)

st.markdown("""##### :red[**This App predicts following Disease:-**]👇 #####""" )

one,two=st.columns((1,1))
one.markdown("""## [**Liver**] ##""")
two.markdown("""## [**Kidney**] ##""")

one.image('https://th.bing.com/th/id/R.44bd49dbc357f02fa2ddb72f3ff7d6d6?rik=w0%2b%2bH2o7KMn3VQ&riu=http%3a%2f%2fniroginepal.com%2fwp-content%2fuploads%2f2016%2f04%2fCirrhosis-of-Liver.png&ehk=9d4MFJTn5uJ4A5qbNQJVlX0eSzJkQ63vE3DQ3n4OTGs%3d&risl=&pid=ImgRaw&r=0',width=280)
two.image('https://static.vecteezy.com/system/resources/previews/044/812/151/non_2x/detailed-human-kidneys-organ-isolated-on-transparent-background-png.png',width=170)

st.markdown("""<br>""",True)
st.markdown("""#### :red[***Select the Disease through sidebar***]  #### """,)

