import os
from dotenv import load_dotenv
import streamlit as st
from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.google import Gemini

#video transcript
from faster_whisper import WhisperModel


def get_transcribed_text(video_file_path: str) -> str:
    model = WhisperModel("base")
    segments, info = model.transcribe(video_file_path)

    transcribed_text=""
    for segment in segments:
        transcribed_text+=segment.text
        print(segment.text)
    
    return transcribed_text


#initialising agent
agent= Agent(
        name = "Video Summarizer Agent",
        model = Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        instructions="Give Consise and short answer",
        markdown= True
    )



st.title("Ai powered Video Q&A System 🎥")  
st.subheader("Powered by GEMINI") 


video_dir = "videos"
if not os.path.exists(video_dir):
    os.makedirs(video_dir)


uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

global trsnscribed_text


if uploaded_file is not None:
    video_path = os.path.join(video_dir, uploaded_file.name)

    # Writing file to "videos" directory
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())


    st.video(video_path)

    trsnscribed_text=get_transcribed_text(video_path)
    #st.write(trsnscribed_text)


else:
    st.write("🤖 Please upload a video.")




if uploaded_file is not None:

    query = st.text_input("🔍 Enter your question about the video:")

    
    if st.button("Submit Query"):
        st.session_state["response"] = ""  # Clear previous response
        if query:
            analysis_prompt = f"""
                You have been given a video, but it has been transcribed into text {trsnscribed_text}. 
                If the user refers to the "video," consider the transcribed text instead.

                **Instructions:**
                - **Do not summarize the video unless explicitly asked.**
                - When referring to the transcribed text, call it "the video" in your response.
                - If the query is **not found** in the video, perform supplementary **web research.

                **User's Query:**
                {query}

                Provide a precise, user-friendly, and actionable response.
            """

            chat_response = agent.run(analysis_prompt)
            st.session_state["response"] = chat_response.content
            st.markdown(st.session_state["response"])
        else:
            st.warning("⚠️ Please enter a query before submitting.")