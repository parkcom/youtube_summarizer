import os
import re
import shutil
import ssl

import openai
import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from pytubefix import YouTube

# SSL 에러 방지
ssl._create_default_https_context = ssl._create_stdlib_context


# 주소를 입력받아 유튜브 동영상의 음성(mp3)을 추출하는 함수
def get_audio(url):
    yt = YouTube(url)
    audio = yt.streams.filter(only_audio=True).first()
    audio_file = audio.download(output_path=".")
    base, ext = os.path.splitext(audio_file)
    new_audio_file = base + ".mp3"
    shutil.move(audio_file, new_audio_file)
    return new_audio_file


# 음성 파일 주소를 전달받아 스크립트를 추출하는 함수
def get_transcript(client, file_path):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file,
        )
    return transcript


# 영어 입력이 들어오면 한글로 번역 및 불릿 포인트 요약을 수행
def trans(client, text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": """당신은 영한 번역가이자 요약가입니다. \
                들어오는 모든 입력을 한국어로 번역하고 불릿 포인트를 이용해 요약해주세요. \
                반드시 불릿 포인트로 요약해야 합니다.
                """,
            },
            {"role": "user", "content": text},
        ],
    )

    return response.choices[0].message.content


# 유튜브 주소의 형태를 정규 표현식으로 체크하는 함수
def youtube_url_check(url):
    pattern = r"^https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]+)(\&ab_channel=[\w\d]+)?$"
    match = re.match(pattern, url)
    return match is not None


def main():
    with st.sidebar:
        openai_apikey = st.text_input(
            "OpenAI API Key", placeholder="Enter your OpenAI API Key", type="password"
        )
        if openai_apikey:
            st.session_state["OPENAI_API_KEY"] = openai_apikey

        st.markdown("---")

    try:
        if st.session_state["OPENAI_API_KEY"]:
            client = openai.OpenAI(api_key=st.session_state["OPENAI_API_KEY"])

    except KeyError:
        st.error("OpenAI API Key를 입력하세요.")
        return

    # session state 초기화
    if "summarize" not in st.session_state:
        st.session_state.summarize = ""

    # 메인 공간
    st.header("Youtube Video Summary & Translation")
    st.image("ai.png", width=200)
    youtube_video_url = st.text_input(
        "Please write down the YouTube address.",
        placeholder="https://www.youtube.com/watch?v=ebjUjzBbj08",
    )
    st.markdown("---")

    if len(youtube_video_url) > 2:
        if not youtube_url_check(youtube_video_url):
            st.error("Please enter a valid YouTube address.")
        else:
            # 동영상 재생 화면 불러오기
            width = 50
            side = width / 2
            _, container, _ = st.columns([side, width, side])
            container.video(data=youtube_video_url)

            # 영상 속 자막 추출하기
            audio_file = get_audio(youtube_video_url)
            transcript = get_transcript(client, audio_file)

            st.subheader("Summary Outcome (in English)")
            llm = ChatOpenAI(
                model="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY")
            )

            # 맵 프롬프트 설정: 1단계 요약에 사용
            prompt = PromptTemplate(
                template="""백틱으로 둘러싸인 전사본을 이용해 해당 유튜브 비디오를 \
                            요약해주세요.
                            ```{text}``` 단, 영어로 작성해 주세요.""",
                input_variables=["text"],
            )

            # 컴바인 프롬프트 설정: 2단계 요약에 사용
            combine_prompt = PromptTemplate(
                template="""백틱으로 둘러싸인 유튜브 스크립트를 모두 조합하여 \
                            ```{text}```
                            10분장 내외의 간결한 요약문을 제공해주세요. 단, 영어로 작성해 주세요. \
""",
                input_variables=["text"],
            )

            # 랭체인을 활용하여 긴 글 요약
            # 긴 문서를 문자열 길이 3000을 기준 길이로 하여 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=3000, chunk_overlap=0
            )

            # 분할된 문서들은 pages라는 문자열 리스트로 저장
            pages = text_splitter.split_text(transcript)
            texts = text_splitter.create_documents(pages)
            chain = load_summarize_chain(
                llm=llm,
                chain_type="map_reduce",
                verbose=False,
                map_prompt=prompt,
                combine_prompt=combine_prompt,
            )

            st.session_state["summarize"] = chain.invoke(texts)["output_text"]
            st.success(st.session_state["summarize"])
            transe = trans(client, st.session_state["summarize"])
            st.subheader("Final Analysis Result (Reply in Korean)")
            st.info(transe)


if __name__ == "__main__":
    main()
