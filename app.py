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
# 1. API 키 및 환경 설정 (Secrets 우선 참조)
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
    st.error("⚠️ OPENAI_API_KEY가 설정되지 않았습니다. .streamlit/secrets.toml을 확인해 주세요.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY, timeout=60.0, max_retries=2)

# ==============================================================================
# 2. PDF 매뉴얼 DB 생성 (HomaBot & 삼성 매뉴얼 동시 적재)
# ==============================================================================
@st.cache_resource(show_spinner="HomaBot 및 삼성 매뉴얼 문서를 읽어 통합 색인표를 구성하는 중입니다...")
def get_vector_store():
    target_filenames = [
        "homabot_manual.pdf",
        "samsung_manual.pdf",
        "삼성로봇청소기 메뉴얼.pdf"
    ]

    all_pages = []
    for fname in target_filenames:
        p = Path(fname)
        if not p.exists() and (Path.cwd().parent / fname).exists():
            p = Path.cwd().parent / fname

        if p.exists():
            loader = PyPDFLoader(str(p))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_name"] = p.name
            all_pages.extend(docs)

    if not all_pages:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(all_pages)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    return FAISS.from_documents(docs, embeddings)

def retrieve_manual_docs(query: str, k: int = 4):
    store = get_vector_store()
    if not store:
        return []
    results = store.similarity_search(query, k=k)
    chunks = []
    for doc in results:
        page_num = doc.metadata.get("page", 0) + 1
        source_name = doc.metadata.get("source_name", "manual.pdf")
        chunks.append({
            "page": page_num,
            "content": doc.page_content,
            "source": source_name
        })
    return chunks

# ==============================================================================
# 3. SerpAPI 구글 웹 검색 함수
# ==============================================================================
def search_web_serpapi(query: str):
    if not SERPAPI_KEY or SERPAPI_KEY.startswith("운영팀에서_"):
        raise RuntimeError("SERPAPI_KEY가 등록되지 않았습니다.")

    params = {
        "q": query.strip(),
        "api_key": SERPAPI_KEY,
        "engine": "google",
        "hl": "ko",
        "gl": "kr",
        "output": "json"
    }
    resp = requests.get("https://serpapi.com/search", params=params, timeout=(5, 30))
    resp.raise_for_status()
    data = resp.json()

    if data.get("error"):
        raise RuntimeError(f"검색 오류: {data.get('error')}")

    results = []
    seen = set()
    for item in data.get("organic_results", []):
        link = (item.get("link") or "").strip()
        title = (item.get("title") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        if not link or link in seen or not (title or snippet):
            continue
        results.append({
            "title": title[:200] or "제목 없음",
            "snippet": snippet[:1000],
            "link": link,
            "date": (item.get("date") or "").strip()
        })
        seen.add(link)
        if len(results) >= 5:
            break
    return results

# ==============================================================================
# 4. LangGraph 그래프 노드 및 조건부 분기
# ==============================================================================
class QAState(TypedDict, total=False):
    question: str
    selected_mode: str
    route: str
    manual_context: list
    search_results: list
    searched_at: str
    answer: str
    error: str

def node_route_question(state: QAState) -> QAState:
    mode = state.get("selected_mode", "자동 선택")
    if mode == "매뉴얼 검색":
        return {"route": "manual"}
    elif mode == "웹 검색":
        return {"route": "web"}
    elif mode == "매뉴얼과 웹 검색 비교":
        return {"route": "compare"}

    q = state.get("question", "").lower()

    compare_kw = [
        "비교", "차이", "차이점", "어느 모델", "어떤 모델", "맞나요", "맞아", 
        "진짜", "사실", "충돌", "다르", "vs", "hx-2", "hx-pro", "공기질"
    ]
    web_kw = [
        "오늘", "현재", "최근", "지금", "실시간", "날씨", "기온", "비", 
        "주가", "환율", "뉴스", "속보", "langgraph", "랭그래프", "serpapi", 
        "가격", "정책", "시세"
    ]
    manual_kw = [
        "homabot", "호마봇", "삼성", "비스포크", "설정", "버튼", "청소", 
        "배터리", "충전", "wi-fi", "wifi", "와이파이", "연결", "필터", 
        "모드", "오류", "에러", "보증", "고객센터", "as", "먼지통", "브러시"
    ]

    if any(k in q for k in compare_kw):
        chosen = "compare"
    elif any(k in q for k in web_kw):
        chosen = "web"
    elif any(k in q for k in manual_kw):
        chosen = "manual"
    else:
        chosen = "manual"

    return {"route": chosen}

def node_manual(state: QAState) -> QAState:
    q = state.get("question", "")
    chunks = retrieve_manual_docs(q)
    return {"manual_context": chunks}

def node_web(state: QAState) -> QAState:
    q = state.get("question", "")
    korea_time = datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")
    try:
        items = search_web_serpapi(q)
        return {"search_results": items, "searched_at": korea_time, "error": ""}
    except Exception as e:
        return {"search_results": [], "searched_at": korea_time, "error": str(e)}

def node_compare(state: QAState) -> QAState:
    q = state.get("question", "")
    korea_time = datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")
    chunks = retrieve_manual_docs(q)
    try:
        items = search_web_serpapi(q)
        return {"manual_context": chunks, "search_results": items, "searched_at": korea_time, "error": ""}
    except Exception as e:
        return {"manual_context": chunks, "search_results": [], "searched_at": korea_time, "error": str(e)}

def node_generate_answer(state: QAState) -> QAState:
    route = state.get("route", "manual")
    q = state.get("question", "")
    manual_docs = state.get("manual_context", [])
    web_docs = state.get("search_results", [])
    err = state.get("error", "")

    if (route in ["web", "compare"]) and err and not web_docs:
        if route == "web":
            return {"answer": f"⚠️ 웹 검색 중 오류가 발생했습니다: {err}\n\nAPI 키 설정 또는 네트워크 상태를 확인해 주세요."}

    context_parts = []
    if manual_docs:
        manual_text = []
        for d in manual_docs:
            manual_text.append(f"[{d['source']} p.{d['page']}] {d['content']}")
        context_parts.append("[매뉴얼 자료]\n" + "\n".join(manual_text))

    if web_docs:
        web_text = []
        for i, w in enumerate(web_docs, start=1):
            web_text.append(f"[{i}] {w['title']} (요약: {w['snippet']})")
        context_parts.append(f"[웹 검색 자료 - 검색시점: {state.get('searched_at', '')}]\n" + "\n".join(web_text))

    system_prompt = (
        "당신은 스마트 로봇청소기 지식 상담 전문가 AI입니다. 아래 주어진 설명서 자료와 웹 자료만을 바탕으로 정직하게 한국어로 답변하세요.\n\n"
        "[핵심 지침]\n"
        "1. 질문 대상 구분: HomaBot 관련 질문인지 삼성 로봇청소기 관련 질문인지 파악하여 해당 설명서 내용을 기반으로 답하세요.\n"
        "2. 자료 충돌 안내: 설명서 내부(예: 모델 비교표와 본문 서술)에서 공기질 모니터링 모델(HX-2 vs HX-PRO) 등 상충되는 내용이 발견되면 "
        "임의로 한쪽을 선택하지 말고 '매뉴얼의 표에서는 HX-2로, 본문에서는 HX-PRO로 기재되어 서로 상충됩니다'와 같이 양쪽 근거를 모두 제시하세요.\n"
        "3. 가상 문서의 한계: homabot_manual.pdf는 실습용 가상 자료이며 출시 연도가 미래로 표기되어 있을 수 있으므로 실제 최신 확정 정보처럼 단정하지 마세요.\n"
        "4. 정보 부족 처리: 설명서에 없는 보증 기간 등의 내용은 '매뉴얼에서 확인할 수 없습니다'라고 솔직히 밝히고 절대 지어내지 마세요.\n"
        "5. 웹 인용: 웹 검색 결과를 인용한 경우 [1], [2] 번호를 문장 끝에 붙이고, 본문 내에 실제 URL 링크는 적지 마세요."
    )

    try:
        res = client.chat.completions.create(
            model=MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"질문: {q}\n\n" + "\n\n".join(context_parts)}
            ]
        )
        base_answer = res.choices[0].message.content.strip()

        footer = []
        if manual_docs:
            source_pages = {}
            for d in manual_docs:
                src = d["source"]
                pg = d["page"]
                if src not in source_pages:
                    source_pages[src] = set()
                source_pages[src].add(pg)

            source_lines = []
            for src, pgs in source_pages.items():
                p_str = ", ".join([f"p.{p}" for p in sorted(list(pgs))])
                source_lines.append(f"`{src}` ({p_str})")
            footer.append("📌 **참조 설명서:** " + " / ".join(source_lines))

        if web_docs:
            links = []
            for i, w in enumerate(web_docs, start=1):
                d_info = f" ({w['date']})" if w['date'] else ""
                links.append(f"**[{i}] {w['title']}**{d_info}  \n{w['link']}")
            footer.append("**검색 자료 출처 (실제 웹페이지 링크):**\n\n" + "\n\n".join(links))

        final_reply = base_answer
        if footer:
            final_reply += "\n\n---\n" + "\n\n".join(footer)

        return {"answer": final_reply}
    except Exception as e:
        return {"answer": f"답변 생성 중 오류가 발생했습니다: {e}"}

def decide_branch(state: QAState) -> str:
    return state.get("route", "manual")

builder = StateGraph(QAState)
builder.add_node("route_question", node_route_question)
builder.add_node("manual_node", node_manual)
builder.add_node("web_node", node_web)
builder.add_node("compare_node", node_compare)
builder.add_node("generate_answer", node_generate_answer)

builder.add_edge(START, "route_question")
builder.add_conditional_edges(
    "route_question",
    decide_branch,
    {
        "manual": "manual_node",
        "web": "web_node",
        "compare": "compare_node"
    }
)
builder.add_edge("manual_node", "generate_answer")
builder.add_edge("web_node", "generate_answer")
builder.add_edge("compare_node", "generate_answer")
builder.add_edge("generate_answer", END)

advanced_bot_app = builder.compile()

# ==============================================================================
# 5. Streamlit 화면 디자인
# ==============================================================================
st.set_page_config(
    page_title="HomaBot 및 삼성 로봇 청소기 지식 상담 챗봇", 
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

# 변경된 메인 제목
st.title("🤖 HomaBot 및 삼성 로봇 청소기 지식 상담 챗봇")
st.caption("HomaBot 및 삼성 로봇청소기 매뉴얼 RAG와 구글 웹 검색을 통합 지원합니다.")

mode_option = st.radio(
    "처리 방식 선택",
    ["🤖 자동 선택", "📘 매뉴얼 검색", "🌐 웹 검색", "⚖️ 매뉴얼과 웹 검색 비교"],
    horizontal=True
)

mode_map = {
    "🤖 자동 선택": "자동 선택",
    "📘 매뉴얼 검색": "매뉴얼 검색",
    "🌐 웹 검색": "웹 검색",
    "⚖️ 매뉴얼과 웹 검색 비교": "매뉴얼과 웹 검색 비교"
}

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

for chat in st.session_state["chat_history"]:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

user_input = st.chat_input("질문을 입력하세요...")

if user_input:
    clean_query = user_input.strip()
    if not clean_query:
        st.warning("⚠️ 질문 글자를 입력해 주세요.")
    else:
        st.session_state["chat_history"].append({"role": "user", "content": clean_query})
        with st.chat_message("user"):
            st.markdown(clean_query)

        with st.chat_message("assistant"):
            with st.spinner("질문 경로를 판별하고 지식을 종합하는 중입니다..."):
                selected_mode_val = mode_map[mode_option]
                result = advanced_bot_app.invoke({
                    "question": clean_query,
                    "selected_mode": selected_mode_val
                })

                final_answer = result.get("answer", "답변을 생성하지 못했습니다.")
                chosen_route = result.get("route", "manual")

                route_badge = {
                    "manual": "📘 매뉴얼 검색",
                    "web": "🌐 웹 검색",
                    "compare": "⚖️ 비교 검색"
                }.get(chosen_route, chosen_route)

                badge_text = f"**사용한 경로:** `{route_badge}`"
                if selected_mode_val == "자동 선택":
                    badge_text += " *(질문 키워드 자동 판별)*"

                full_display = f"{badge_text}\n\n{final_answer}"
                st.markdown(full_display)
                st.session_state["chat_history"].append({"role": "assistant", "content": full_display})
