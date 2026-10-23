
import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="어린이보호구역 교통안전 분석",
    page_icon="🚸",
    layout="wide",
)

st.title("🚸 어린이보호구역 교통안전 분석")
st.markdown("""
2022년부터 2024년까지의 데이터를 이용해
어린이보호구역 지정 현황과 교통사고 변화를 분석합니다.
""")

# -----------------------------
# 데이터
# -----------------------------
years = [2022, 2023, 2024]

zones = pd.DataFrame({
    "연도": years,
    "어린이보호구역 지정 수": [16641, 16490, 16308],
})

traffic = pd.DataFrame({
    "연도": years,
    "사고 건수": [514, 486, 526],
    "사망자 수": [3, 2, 2],
    "부상자 수": [529, 523, 556],
})

# -----------------------------
# 핵심 수치
# -----------------------------
st.header("📌 핵심 데이터")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "2024년 보호구역 지정 수",
        f"{zones.iloc[-1]['어린이보호구역 지정 수']:,}개",
        delta=f"{zones.iloc[-1]['어린이보호구역 지정 수'] - zones.iloc[0]['어린이보호구역 지정 수']:,}개 (2022년 대비)",
    )

with col2:
    st.metric(
        "2024년 교통사고",
        f"{traffic.iloc[-1]['사고 건수']:,}건",
        delta=f"{traffic.iloc[-1]['사고 건수'] - traffic.iloc[-2]['사고 건수']:+,}건 (전년 대비)",
    )

with col3:
    st.metric(
        "2024년 부상자",
        f"{traffic.iloc[-1]['부상자 수']:,}명",
        delta=f"{traffic.iloc[-1]['부상자 수'] - traffic.iloc[-2]['부상자 수']:+,}명 (전년 대비)",
    )

st.divider()

# -----------------------------
# 그래프 1: 보호구역 지정 현황
# -----------------------------
st.header("1. 어린이보호구역 지정 현황")

fig1 = px.bar(
    zones,
    x="연도",
    y="어린이보호구역 지정 수",
    text="어린이보호구역 지정 수",
    title="연도별 어린이보호구역 지정 현황",
    labels={
        "연도": "연도",
        "어린이보호구역 지정 수": "지정 구역 수(개)",
    },
)

fig1.update_traces(
    texttemplate="%{text:,}",
    textposition="outside",
)
fig1.update_layout(xaxis=dict(type="category"))

st.plotly_chart(fig1, use_container_width=True)

st.caption(
    "2022년 16,641개 → 2023년 16,490개 → 2024년 16,308개"
)

st.divider()

# -----------------------------
# 그래프 2: 교통사고 건수
# -----------------------------
st.header("2. 어린이보호구역 내 교통사고 건수")

fig2 = px.bar(
    traffic,
    x="연도",
    y="사고 건수",
    text="사고 건수",
    title="연도별 교통사고 건수",
    labels={
        "연도": "연도",
        "사고 건수": "사고 건수(건)",
    },
)

fig2.update_traces(textposition="outside")
fig2.update_layout(xaxis=dict(type="category"))

st.plotly_chart(fig2, use_container_width=True)

st.divider()

# -----------------------------
# 그래프 3: 사망자 수
# -----------------------------
st.header("3. 어린이보호구역 내 사망자 수")

fig3 = px.bar(
    traffic,
    x="연도",
    y="사망자 수",
    text="사망자 수",
    title="연도별 사망자 수",
    labels={
        "연도": "연도",
        "사망자 수": "사망자 수(명)",
    },
)

fig3.update_traces(textposition="outside")
fig3.update_layout(xaxis=dict(type="category"))

st.plotly_chart(fig3, use_container_width=True)

st.divider()

# -----------------------------
# 그래프 4: 부상자 수
# -----------------------------
st.header("4. 어린이보호구역 내 부상자 수")

fig4 = px.bar(
    traffic,
    x="연도",
    y="부상자 수",
    text="부상자 수",
    title="연도별 부상자 수",
    labels={
        "연도": "연도",
        "부상자 수": "부상자 수(명)",
    },
)

fig4.update_traces(textposition="outside")
fig4.update_layout(xaxis=dict(type="category"))

st.plotly_chart(fig4, use_container_width=True)

st.divider()

# -----------------------------
# 그래프 5: 사고 건수와 부상자 수 비교
# -----------------------------
st.header("5. 교통사고와 부상자 수 변화 비교")

traffic_long = traffic.melt(
    id_vars="연도",
    value_vars=["사고 건수", "부상자 수"],
    var_name="항목",
    value_name="건수 또는 인원",
)

fig5 = px.line(
    traffic_long,
    x="연도",
    y="건수 또는 인원",
    color="항목",
    markers=True,
    text="건수 또는 인원",
    title="연도별 사고 건수와 부상자 수",
    labels={
        "연도": "연도",
        "건수 또는 인원": "건수 또는 명",
        "항목": "구분",
    },
)

fig5.update_traces(textposition="top center")
fig5.update_layout(xaxis=dict(type="category"))

st.plotly_chart(fig5, use_container_width=True)

st.divider()

# -----------------------------
# 데이터 표
# -----------------------------
st.header("📋 데이터 표")

with st.expander("어린이보호구역 지정 현황 보기"):
    st.dataframe(
        zones,
        hide_index=True,
        use_container_width=True,
    )

with st.expander("교통사고 현황 보기"):
    st.dataframe(
        traffic,
        hide_index=True,
        use_container_width=True,
    )

st.divider()

# -----------------------------
# 데이터 분석
# -----------------------------
st.header("🔍 데이터에서 발견한 점")

st.subheader("1. 보호구역 지정 수는 감소했다")
st.write(
    "어린이보호구역 지정 수는 2022년 16,641개에서 "
    "2024년 16,308개로 333개 감소했습니다."
)

st.subheader("2. 교통사고는 2024년에 증가했다")
st.write(
    "사고 건수는 2022년 514건, 2023년 486건, "
    "2024년 526건으로 나타났습니다. "
    "2023년에 감소했지만 2024년에는 다시 증가했습니다."
)

st.subheader("3. 부상자 수는 2024년에 가장 많았다")
st.write(
    "부상자 수는 2022년 529명, 2023년 523명, "
    "2024년 556명으로 나타났습니다."
)

st.subheader("4. 사망자 수는 감소한 수준을 유지했다")
st.write(
    "사망자 수는 2022년 3명에서 2023년 2명으로 감소했고, "
    "2024년에도 2명으로 유지되었습니다."
)

st.warning(
    "주의: 이 데이터만으로 보호구역 지정 수의 감소가 "
    "교통사고 증가의 원인이라고 단정할 수 없습니다. "
    "차량 통행량, 보호구역별 사고율, 안전시설 변화 등의 "
    "자료를 함께 분석해야 합니다."
)

st.divider()

# -----------------------------
# 탐구 질문
# -----------------------------
st.header("💡 더 탐구해 볼 질문")

st.markdown("""
- 어린이보호구역 지정 수가 줄어든 이유는 무엇일까?
- 보호구역의 안전시설을 개선하면 사고를 줄일 수 있을까?
- 사고 건수뿐 아니라 차량 통행량을 고려하면 결과가 달라질까?
- 어린이 교통사고를 줄이기 위해 어떤 정책이 필요할까?
""")

st.caption("자료: 사용자가 제공한 2022~2024년 요약 데이터")
