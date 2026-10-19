import os
import json
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from langgraph.graph import StateGraph, START, END

# ==============================================================================
# 1. 안전하게 비밀 열쇠(API 키) 꺼내오기
# ==============================================================================
load_dotenv()

def read_secret(key_name, default=""):
    if key_name in st.secrets:
        return str(st.secrets[key_name]).strip()
    return os.getenv(key_name, default).strip()

OPENAI_API_KEY = read_secret("OPENAI_API_KEY")
SERPAPI_KEY = read_secret("SERPAPI_KEY") or read_secret("SERPAPI_API_KEY")
MODEL = read_secret("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("운영팀에서_"):
    st.error("⚠️ OPENAI_API_KEY가 없습니다. .streamlit/secrets.toml 파일을 확인해 주세요.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY, timeout=60.0, max_retries=2)

# ==============================================================================
# 2. PDF 매뉴얼 책 읽기 및 검색 (RAG 기능)
# ==============================================================================
@st.cache_resource(show_spinner="HomaBot 설명서 책을 읽어서 색인표를 만드는 중입니다...")
def make_manual_database():
    pdf_file = Path("homabot_manual.pdf")
    if not pdf_file.exists() and (Path.cwd().parent / "homabot_manual.pdf").exists():
        pdf_file = Path.cwd().parent / "homabot_manual.pdf"

    if not pdf_file.exists():
        return None

    pages = PyPDFLoader(str(pdf_file)).load()
    text_cutter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    small_chunks = text_cutter.split_documents(pages)

    magic_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    digital_library = FAISS.from_documents(small_chunks, magic_embeddings)
    return digital_library

def answer_from_manual(user_question: str):
    digital_library = make_manual_database()
    if digital_library is None:
        return "[오류] homabot_manual.pdf 파일이 없습니다. 프로젝트 폴더에 넣어주세요.", []

    best_chunks = digital_library.similarity_search(user_question, k=3)
    if not best_chunks:
        return "설명서에서 관련된 내용을 찾지 못했습니다.", []

    reference_texts = []
    found_pages = set()
    for chunk in best_chunks:
        page_num = chunk.metadata.get("page", 0) + 1
        found_pages.add(page_num)
        reference_texts.append(f"--- [참고 {page_num}페이지] ---\n" + chunk.page_content)

    rules = (
        "당신은 HomaBot 제품 안내원입니다. 아래 설명서 내용에 있는 사실만 친절하게 답하세요.\n"
        "설명서에 없는 가격이나 사양은 마음대로 지어내지 말고 고객센터로 문의하라고 안내하세요.\n\n"
        f"[설명서 내용]\n" + "\n\n".join(reference_texts)
    )

    try:
        reply = client.chat.completions.create(
            model=MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": rules},
                {"role": "user", "content": user_question}
            ]
        )
        return reply.choices[0].message.content, sorted(list(found_pages))
    except OpenAIError as e:
        return f"인공지능 답변 실패: {e}", []

# ==============================================================================
# 3. 구글 웹 검색 (SerpAPI + LangGraph 기능)
# ==============================================================================
class WebQAState(TypedDict, total=False):
    question: str
    search_results: list
    searched_at: str
    answer: str
    error: str

def do_google_search(query_text: str):
    if not SERPAPI_KEY or SERPAPI_KEY.startswith("운영팀에서_"):
        raise RuntimeError("SERPAPI_KEY가 등록되지 않았습니다.")

    search_params = {
        "q": query_text.strip(),
        "api_key": SERPAPI_KEY,
        "engine": "google",
        "hl": "ko",
        "gl": "kr",
        "output": "json"
    }
    response = requests.get("https://serpapi.com/search", params=search_params, timeout=(5, 30))
    response.raise_for_status()
    result_data = response.json()

    if result_data.get("error"):
        raise RuntimeError("검색 실패: " + str(result_data.get("error")))

    clean_list = []
    already_seen_links = set()
    for doc in result_data.get("organic_results", []):
        link = (doc.get("link") or "").strip()
        title = (doc.get("title") or "").strip()
        snippet = (doc.get("snippet") or "").strip()
        if not link or link in already_seen_links or not (title or snippet):
            continue
        clean_list.append({
            "title": title[:200] or "제목 없음",
            "snippet": snippet[:1000],
            "link": link,
            "date": (doc.get("date") or "").strip()
        })
        already_seen_links.add(link)
        if len(clean_list) >= 5:
            break
    return clean_list

def step_search(state: WebQAState) -> WebQAState:
    q = state.get("question", "").strip()
    korea_time = datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")
    try:
        items = do_google_search(q)
        return {"question": q, "search_results": items, "searched_at": korea_time, "error": ""}
    except Exception as err:
        return {"question": q, "search_results": [], "searched_at": korea_time, "error": str(err)}

def step_write_answer(state: WebQAState) -> WebQAState:
    if state.get("error"):
        return {"answer": f"[검색 실패] {state['error']}"}

    docs = state.get("search_results", [])
    if not docs:
        return {"answer": "사용할 수 있는 검색 결과가 없습니다. 질문을 더 구체적으로 적어주세요."}

    doc_numbered = [{"번호": i, **item} for i, item in enumerate(docs, start=1)]
    bundle_text = json.dumps({
        "question": state["question"],
        "searched_at": state["searched_at"],
        "search_results": doc_numbered
    }, ensure_ascii=False)

    prompt_rules = (
        "당신은 구글 검색 자료를 바탕으로 한국어로 성실히 답변하는 인공지능입니다.\n"
        "1. 제공된 제목과 요약글에 실제로 적힌 내용만 답하세요.\n"
        "2. 없는 내용을 추측해서 거짓말로 지어내지 마세요.\n"
        "3. 근거 문장마다 [1], [2]처럼 출처 번호를 반드시 붙여주세요.\n"
        "4. 본문 안에 인터넷 주소(URL)는 직접 적지 마세요."
    )

    try:
        ai_resp = client.chat.completions.create(
            model=MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": prompt_rules},
                {"role": "user", "content": bundle_text}
            ]
        )
        final_text = ai_resp.choices[0].message.content.strip()

        # 출처 링크 줄바꿈 처리
        link_list_text = []
        for i, item in enumerate(docs, start=1):
            date_info = item["date"] if item["date"] else "작성일 미제공"
            link_list_text.append(f"**[{i}] {item['title']}** ({date_info})  \n{item['link']}")

        complete_reply = final_text + "\n\n**검색 자료 출처 (실제 웹페이지 링크):**\n\n" + "\n\n".join(link_list_text)
        return {"answer": complete_reply}
    except Exception as err:
        return {"answer": f"[답변 생성 오류]: {err}"}

graph_builder = StateGraph(WebQAState)
graph_builder.add_node("search_node", step_search)
graph_builder.add_node("writer_node", step_write_answer)
graph_builder.add_edge(START, "search_node")
graph_builder.add_edge("search_node", "writer_node")
graph_builder.add_edge("writer_node", END)
web_chatbot_app = graph_builder.compile()

# ==============================================================================
# 4. Streamlit 화면 디자인 (중앙 정렬 + 900px 너비 최적화)
# ==============================================================================
st.set_page_config(
    page_title="HomaBot 안내 챗봇",
    page_icon="🤖",
    layout="centered"
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 900px !important;
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        margin: 0 auto !important;
    }
    h1 {
        white-space: nowrap !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🤖 HomaBot 설명서 & 구글 검색 통합 챗봇")
st.caption("궁금한 방식을 선택하고 아래에 질문을 입력해 보세요!")

search_mode = st.radio(
    "어디서 찾을까요?",
    ["매뉴얼에서 찾기", "웹에서 찾기"],
    horizontal=True
)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

for chat in st.session_state["chat_history"]:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

user_input = st.chat_input("질문을 입력하세요...")

if user_input:
    clean_query = user_input.strip()

    if not clean_query:
        st.warning("질문 글자를 입력해 주세요.")
    else:
        st.session_state["chat_history"].append({"role": "user", "content": clean_query})
        with st.chat_message("user"):
            st.markdown(clean_query)

        with st.chat_message("assistant"):
            with st.spinner("자료를 찾고 답변을 만드는 중입니다..."):
                if search_mode == "매뉴얼에서 찾기":
                    bot_text, pages = answer_from_manual(clean_query)
                    if pages:
                        pages_str = ", ".join([f"p.{p}" for p in pages])
                        final_output = f"{bot_text}\n\n📌 **참조 설명서 페이지:** `homabot_manual.pdf` ({pages_str})"
                    else:
                        final_output = bot_text
                else:
                    graph_result = web_chatbot_app.invoke({"question": clean_query})
                    final_output = graph_result.get("answer", "답변을 가져오지 못했습니다.")

                st.markdown(final_output)
                st.session_state["chat_history"].append({"role": "assistant", "content": final_output})
