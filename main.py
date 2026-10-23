
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="어린이보호구역 교통안전 분석",
    page_icon="🚸",
    layout="wide",
)

st.title("🚸 어린이보호구역 교통안전 분석")
st.write("2022~2024년 어린이보호구역 지정 현황과 교통사고 데이터를 살펴봅니다.")

# 데이터
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

# 1. 어린이보호구역 지정 현황
st.header("1. 어린이보호구역 지정 현황")

fig_zones = px.bar(
    zones,
    x="연도",
    y="어린이보호구역 지정 수",
    text="어린이보호구역 지정 수",
    title="연도별 어린이보호구역 지정 현황",
    labels={
        "연도": "연도",
        "어린이보호구역 지정 수": "지정 구역 수"
    },
)

fig_zones.update_traces(
    texttemplate="%{text:,}",
    textposition="outside"
)
fig_zones.update_layout(xaxis=dict(type="category"))

st.plotly_chart(fig_zones, use_container_width=True)
st.dataframe(zones, hide_index=True, use_container_width=True)

# 2. 교통사고 및 부상자 수
st.header("2. 어린이보호구역 내 교통사고 및 부상자 수")

traffic_long = traffic.melt(
    id_vars="연도",
    value_vars=["사고 건수", "부상자 수"],
    var_name="항목",
    value_name="건수 또는 인원",
)

fig_traffic = px.line(
    traffic_long,
    x="연도",
    y="건수 또는 인원",
    color="항목",
    markers=True,
    text="건수 또는 인원",
    title="연도별 교통사고 및 부상자 수",
    labels={
        "연도": "연도",
        "건수 또는 인원": "건수 또는 명",
        "항목": "항목"
    },
)

fig_traffic.update_traces(textposition="top center")
fig_traffic.update_layout(xaxis=dict(type="category"))

st.plotly_chart(fig_traffic, use_container_width=True)

# 3. 사망자 수
st.header("3. 어린이보호구역 내 사망자 수")

fig_deaths = px.bar(
    traffic,
    x="연도",
    y="사망자 수",
    text="사망자 수",
    title="연도별 사망자 수",
    labels={
        "연도": "연도",
        "사망자 수": "사망자 수(명)"
    },
)

fig_deaths.update_traces(textposition="outside")
fig_deaths.update_layout(xaxis=dict(type="category"))

st.plotly_chart(fig_deaths, use_container_width=True)

# 4. 데이터 요약
st.header("4. 데이터 요약")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "2024년 보호구역 수",
        f"{zones.iloc[-1]['어린이보호구역 지정 수']:,}개"
    )

with col2:
    st.metric(
        "2024년 사고 건수",
        f"{traffic.iloc[-1]['사고 건수']:,}건"
    )

with col3:
    st.metric(
        "2024년 부상자 수",
        f"{traffic.iloc[-1]['부상자 수']:,}명"
    )

# 5. 데이터 해석
st.header("5. 데이터 분석 결과")

st.write("""
- 어린이보호구역 지정 수는 2022년 16,641개에서
  2024년 16,308개로 감소했습니다.
- 교통사고 건수는 2023년 486건에서
  2024년 526건으로 증가했습니다.
- 부상자 수는 2024년 556명으로
  최근 3년 중 가장 많았습니다.
- 사망자 수는 2022년 3명에서 2023년과
  2024년에는 각각 2명이었습니다.
""")

st.warning(
    "이 자료만으로 보호구역 지정 수 감소가 "
    "교통사고 증가의 원인이라고 단정할 수 없습니다. "
    "차량 통행량과 안전시설 등의 자료도 함께 살펴보세요."
)

st.caption("자료: 제공된 2022~2024년 요약 데이터")
